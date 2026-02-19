import base64
import logging
import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from backend.app.core.rag_pipeline import (
    detect_domain,
    embed_image,
    embed_text,
    retrieve_adaptive_k,
    qwen3vl_generate,
)
from backend.app.services import chat_repository

try:
    from backend.app.services.chat_storage import store_attachment
except Exception:  # pragma: no cover - optional storage dependency
    def store_attachment(*_args, **_kwargs):  # type: ignore
        raise RuntimeError("Supabase storage is not configured")

log = logging.getLogger("medrag.langchain")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", Path(os.getcwd()) / "uploads"))


@dataclass
class SessionState:
    memory: ConversationBufferMemory
    image_history: List[str] = field(default_factory=list)
    context_snapshots: List[str] = field(default_factory=list)
    remote_cache: Dict[str, str] = field(default_factory=dict)
    temp_files: List[str] = field(default_factory=list)
    hidden_turns: List[int] = field(default_factory=list)
    initial_summary: Optional[str] = None
    initial_summary_used: bool = False


class LangChainChatOrchestrator:
    """LangChain-based orchestrator layering conversation memory over MED-RAG."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._model_name = os.getenv("QWEN_MODEL", "qwen/qwen3-vl-235b-a22b-thinking")
        self._llm: Optional[ChatOpenAI] = None
        self._refresh_openrouter_client()
        self._initial_summary_question = (
            "Provide baseline Findings and Impression for the uploaded case solely from the"
            " retrieved context. Mention uncertainty if evidence is limited."
        )

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are MED-RAG, a cautious medical imaging assistant. Use the retrieved"
                    " context, previously discussed findings, and referenced images to craft"
                    " accurate, concise answers. State uncertainty when present and avoid"
                    " hallucinations. {style_directive}",
                ),
                MessagesPlaceholder("history"),
                (
                    "human",
                    "Question: {question}\n\nKnown session image references:\n{image_memory}\n\n"
                    "Initial findings (first pass):\n{initial_summary}\n\n"
                    "Retrieved context:\n{context_block}\n\nDomain hint: {domain}",
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def _hydrate_history(self, session_id: str, state: SessionState) -> None:
        """Populate in-memory history from persisted messages."""
        try:
            persisted = chat_repository.load_messages(session_id)
        except Exception as exc:
            log.exception("Failed to load messages from Supabase for %s: %s", session_id, exc)
            return

        if not persisted:
            return

        for message in persisted:
            role = message.get("role")
            content = message.get("content", "")
            attachment_url = message.get("attachment_url")
            if role == "user":
                state.memory.save_context({"question": content}, {"answer": ""})
            elif role == "assistant":
                state.memory.save_context({"question": ""}, {"answer": content})
            else:
                continue

            # Post-process to trim empty placeholders introduced by save_context
            history = state.memory.load_memory_variables({}).get("history", [])
            if history and history[-1].type == "ai" and not history[-1].content:
                history.pop()

            if role == "user" and attachment_url:
                state.image_history.append(attachment_url)

    def _ensure_session(self, session_id: str, user_id: Optional[str] = None) -> None:
        if session_id in self._sessions:
            return

        memory = ConversationBufferMemory(
            memory_key="history",
            input_key="question",
            output_key="answer",
            return_messages=True,
        )
        state = SessionState(memory=memory)
        self._sessions[session_id] = state

        existing = chat_repository.get_session(session_id)
        if not existing:
            if not user_id:
                return
            chat_repository.create_session(user_id=user_id, session_id=session_id)
        self._hydrate_history(session_id, state)

    def _get_session(self, session_id: str, user_id: Optional[str] = None) -> SessionState:
        if not session_id:
            raise ValueError("session_id is required")
        self._ensure_session(session_id, user_id=user_id)
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
        # Optionally delete persisted session to keep storage in sync
        try:
            chat_repository.delete_session(session_id)
        except Exception as exc:
            log.warning("Failed to delete session %s from Supabase: %s", session_id, exc)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            persisted = chat_repository.load_messages(session_id)
        except Exception as exc:
            log.exception("Failed to load history for session %s: %s", session_id, exc)
            persisted = []

        if not persisted:
            state = self._get_session(session_id)
            history = state.memory.load_memory_variables({}).get("history", [])
            hidden = set(state.hidden_turns)
            return [
                {"type": msg.type, "content": msg.content}
                for idx, msg in enumerate(history)
                if idx not in hidden
            ]

        history_payload: List[Dict[str, Any]] = []
        for item in persisted:
            role = item.get("role")
            msg_type = "human" if role == "user" else "ai"
            history_payload.append(
                {
                    "type": msg_type,
                    "content": item.get("content", ""),
                    "attachment_url": item.get("attachment_url"),
                    "created_at": item.get("created_at"),
                }
            )
        return history_payload

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _openrouter_headers(self) -> Dict[str, str]:
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
        title = os.getenv("OPENROUTER_APP_TITLE", "MED-RAG (Local Dev)")
        headers: Dict[str, str] = {"X-Title": title}
        if referer:
            headers["HTTP-Referer"] = referer
        return headers

    def _refresh_openrouter_client(self) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not api_key:
            log.warning("OPENROUTER_API_KEY / OPENAI_API_KEY missing – LangChain LLM disabled.")
            self._llm = None
            return
        try:
            self._llm = ChatOpenAI(
                model=self._model_name,
                temperature=0.2,
                max_tokens=600,
                openai_api_key=api_key,
                openai_api_base=base_url,
                default_headers=self._openrouter_headers(),
            )
        except Exception as exc:
            log.exception("Failed to initialise ChatOpenAI client: %s", exc)
            self._llm = None

    # ------------------------------------------------------------------
    # Core orchestration
    # ------------------------------------------------------------------
    def _format_context(self, retrieved: List[Dict[str, Any]]) -> str:
        if not retrieved:
            return "(No retrieved context available – reply using prior knowledge and clarify." \
                " that retrieval returned nothing.)"
        blocks: List[str] = []
        for idx, item in enumerate(retrieved, 1):
            snippet = (item.get("report_text") or "").strip()
            snippet = snippet[:500]
            image_path = item.get("image_path") or item.get("projection")
            cert = item.get("certainty")
            hybrid = item.get("hybrid_score")
            header = f"Case {idx}"
            meta_parts = []
            if cert is not None:
                meta_parts.append(f"certainty={cert:.2f}")
            if hybrid is not None:
                meta_parts.append(f"hybrid={hybrid:.2f}")
            if image_path:
                meta_parts.append(f"image={image_path}")
            if meta_parts:
                header += " (" + ", ".join(meta_parts) + ")"
            blocks.append(f"{header}:\n{snippet or '[no report text stored]'}")
        return "\n\n".join(blocks)

    def _needs_clarification(self, question: str, retrieved: List[Dict[str, Any]]) -> bool:
        if not question or not question.strip():
            return False
        if not retrieved:
            return True

        cert_scores = [item.get("certainty") for item in retrieved if item.get("certainty") is not None]
        hybrid_scores = [item.get("hybrid_score") for item in retrieved if item.get("hybrid_score") is not None]

        top_certainty = max(cert_scores) if cert_scores else 0.0
        top_hybrid = max(hybrid_scores) if hybrid_scores else 0.0

        return top_certainty < 0.45 and top_hybrid < 0.45

    _IMAGE_QUERY_PATTERNS = [
        "this image",
        "the image",
        "this scan",
        "the scan",
        "this xray",
        "this x-ray",
        "the xray",
        "the x-ray",
        "this radiograph",
        "the radiograph",
        "this photo",
        "the photo",
        "this picture",
        "the picture",
        "this fundus",
        "the fundus",
        "does this show",
        "what does this",
        "findings in this",
    ]

    _IMAGE_KEYWORD_REGEX = re.compile(r"\b(image|scan|x-?ray|radiograph|fundus|slide|photo|picture)\b", re.IGNORECASE)

    def _question_targets_image(self, question: str) -> bool:
        if not question:
            return False
        lowered = question.lower()
        if any(pattern in lowered for pattern in self._IMAGE_QUERY_PATTERNS):
            return True
        if "case" in lowered and ("this" in lowered or "the" in lowered):
            if "case" in lowered and ("describe" in lowered or "findings" in lowered):
                return True
        if "uploaded" in lowered and self._IMAGE_KEYWORD_REGEX.search(lowered):
            return True
        # standalone keyword without context is too generic; require question to start with directive referencing image
        if lowered.startswith("describe the") and self._IMAGE_KEYWORD_REGEX.search(lowered):
            return True
        if lowered.startswith("what does the") and self._IMAGE_KEYWORD_REGEX.search(lowered):
            return True
        return False

    def _resolve_image_input(self, supplied_path: Optional[str], state: SessionState) -> Optional[str]:
        if not supplied_path:
            return None

        parsed = urlparse(supplied_path)

        # handle already-downloaded cache
        if supplied_path in state.remote_cache:
            cached_path = state.remote_cache[supplied_path]
            if cached_path and Path(cached_path).exists():
                return cached_path

        # URLs that point to our uploads directory can be mapped directly
        if parsed.scheme in {"http", "https"}:
            if parsed.path.startswith("/uploads/"):
                candidate = UPLOAD_DIR / Path(parsed.path).name
                if candidate.exists():
                    state.remote_cache[supplied_path] = str(candidate)
                    return str(candidate)

            # fall back to downloading
            try:
                resp = requests.get(supplied_path, timeout=20)
                resp.raise_for_status()
                suffix = Path(parsed.path).suffix or ".png"
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp_file.write(resp.content)
                tmp_file.flush()
                tmp_file.close()
                state.remote_cache[supplied_path] = tmp_file.name
                state.temp_files.append(tmp_file.name)
                return tmp_file.name
            except Exception as exc:
                log.warning("Failed to download image %s: %s", supplied_path, exc)
                return None

        # handle local or relative paths
        candidate = Path(supplied_path)
        if candidate.exists():
            return str(candidate.resolve())

        if not candidate.is_absolute():
            upload_candidate = UPLOAD_DIR / candidate
            if upload_candidate.exists():
                return str(upload_candidate.resolve())

        log.warning("Image path %s could not be resolved.", supplied_path)
        return None

    @staticmethod
    def _image_to_data_url(image_path: str) -> Optional[str]:
        try:
            mime, _ = mimetypes.guess_type(image_path)
            mime = mime or "image/png"
            with open(image_path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        except Exception as exc:
            log.warning("Failed to serialise image %s as data URL: %s", image_path, exc)
            return None

    def _retrieve_context(
        self,
        question: str,
        image_path: Optional[str],
        state: SessionState,
    ) -> Dict[str, Any]:
        supplied = image_path or (state.image_history[-1] if state.image_history else None)
        final_image = self._resolve_image_input(supplied, state)

        domain_source = final_image if final_image else image_path
        domain_info = detect_domain(question, domain_source)

        modality = "text"
        if final_image:
            try:
                query_vec = embed_image(final_image)
                modality = "image"
            except FileNotFoundError:
                log.warning("Image %s not found on disk; falling back to text embedding.", final_image)
                query_vec = embed_text(question)
                modality = "text"
        else:
            query_vec = embed_text(question)

        domain_filter = domain_info.get("domain") if domain_info else None
        retrieved_objs = retrieve_adaptive_k(query_vec, modality=modality, domain_filter=domain_filter)
        retrieved_context: List[Dict[str, Any]] = []
        for obj in retrieved_objs:
            props = obj.get("properties", obj)
            retrieved_context.append(
                {
                    "report_text": props.get("report_text", ""),
                    "image_path": props.get("image_path") or props.get("projection"),
                    "left_image_path": props.get("left_image_path"),
                    "right_image_path": props.get("right_image_path"),
                    "projection": props.get("projection"),
                    "certainty": float(obj.get("_additional", {}).get("certainty", 0.0)),
                    "hybrid_score": float(obj.get("_hybrid_score", 0.0)),
                    "ophthalmology_labels": props.get("ophthalmology_labels"),
                }
            )
        context_block = self._format_context(retrieved_context)
        return {
            "domain_info": domain_info,
            "context_block": context_block,
            "retrieved": retrieved_context,
            "used_image": final_image,
        }

    def _generate_initial_summary(
        self,
        state: SessionState,
        context: Dict[str, Any],
        domain: str,
    ) -> None:
        if self._llm is None:
            log.warning("Cannot generate initial summary because LLM client is not configured.")
            state.initial_summary = None
            state.initial_summary_used = False
            return

        history_messages = state.memory.load_memory_variables({}).get("history", [])
        image_memory = "\n".join(state.image_history) if state.image_history else "(none yet)"

        messages = self._prompt.format_prompt(
            history=history_messages,
            question=self._initial_summary_question,
            context_block=context["context_block"],
            initial_summary="(no prior findings captured)",
            domain=domain,
            image_memory=image_memory,
            style_directive="Structure the reply with 'Findings' and 'Impression' sections grounded in the retrieved cases.",
        ).to_messages()

        try:
            response = self._llm.invoke(messages)
            summary_text = response.content.strip()
        except Exception as exc:
            log.exception("Failed to generate initial summary: %s", exc)
            summary_text = "Initial findings unavailable due to generation error."

        before_len = len(history_messages)
        state.memory.save_context(
            {"question": self._initial_summary_question},
            {"answer": summary_text},
        )
        updated_history = state.memory.load_memory_variables({}).get("history", [])
        after_len = len(updated_history)
        if after_len > before_len:
            state.hidden_turns.extend(range(before_len, after_len))

        state.initial_summary = summary_text
        state.initial_summary_used = False

    def generate_response(
        self,
        session_id: str,
        question: str,
        image_path: Optional[str] = None,
        domain_hint: Optional[str] = None,
        silent: bool = False,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not question and not image_path:
            raise ValueError("Provide a question or an image for the assistant to reason about.")

        state = self._get_session(session_id, user_id=user_id)

        if image_path and image_path not in state.image_history:
            state.image_history.append(image_path)

        image_question = self._question_targets_image(question)
        should_use_image_context = image_question and (image_path or state.image_history)

        if should_use_image_context:
            context = self._retrieve_context(question, image_path, state)
            if context["context_block"]:
                state.context_snapshots.append(context["context_block"])
        else:
            domain_info = detect_domain(question, None)
            context = {
                "domain_info": domain_info or {},
                "context_block": "",
                "retrieved": [],
                "used_image": None,
            }

        domain = domain_hint or context["domain_info"].get("domain", "radiology")

        if (
            not silent
            and state.initial_summary is None
            and context["retrieved"]
        ):
            self._generate_initial_summary(state, context, domain)

        if (
            not silent
            and should_use_image_context
            and self._needs_clarification(question, context["retrieved"])
        ):
            clarification = (
                "I couldn't find context in the uploaded image that matches that question. "
                "Could you clarify how your question relates to this case?"
            )
            state.memory.save_context({"question": question}, {"answer": clarification})
            return {
                "answer": clarification,
                "domain": domain,
                "domain_info": context["domain_info"],
                "retrieved": context["retrieved"],
                "used_image": context["used_image"],
                "history": self.get_history(session_id),
                "memory_disabled": False,
            }

        history_messages = state.memory.load_memory_variables({}).get("history", [])
        image_memory = "\n".join(state.image_history) if should_use_image_context else "(none relevant)"
        if state.initial_summary and not state.initial_summary_used and should_use_image_context:
            initial_summary_block = state.initial_summary
        else:
            initial_summary_block = ""

        if should_use_image_context:
            context_block_to_use = context["context_block"]
            style_directive = (
                "When appropriate, organise your answer with 'Findings' and 'Impression' headings grounded"
                " in the retrieved cases and cached initial summary."
            )
        else:
            context_block_to_use = ""
            style_directive = (
                "Provide a comprehensive answer grounded in established medical knowledge. Do not refer"
                " to retrieved context, images, hidden findings, or the fact that you are relying on"
                " general knowledge. Avoid headings such as 'Findings' or 'Impression' unless the user"
                " explicitly requests them."
            )

        messages = self._prompt.format_prompt(
            history=history_messages,
            question=question,
            context_block=context_block_to_use,
            initial_summary=initial_summary_block,
            domain=domain,
            image_memory=image_memory,
            style_directive=style_directive,
        ).to_messages()

        if should_use_image_context and context.get("used_image"):
            data_url = self._image_to_data_url(context["used_image"])
            if data_url:
                human_msg = messages[-1]
                image_part = {"type": "image_url", "image_url": {"url": data_url}}
                if isinstance(human_msg.content, str):
                    human_msg.content = [
                        {"type": "text", "text": human_msg.content},
                        image_part,
                    ]
                elif isinstance(human_msg.content, list):
                    normalised_content = []
                    for item in human_msg.content:
                        if isinstance(item, str):
                            normalised_content.append({"type": "text", "text": item})
                        else:
                            normalised_content.append(item)
                    normalised_content.append(image_part)
                    human_msg.content = normalised_content

        if silent:
            if self._llm is None:
                log.warning("Silent prime requested but LLM client is not configured.")
                return {
                    "primed": True,
                    "domain": domain,
                    "domain_info": context["domain_info"],
                    "used_image": context["used_image"],
                }

            try:
                response = self._llm.invoke(messages)
                answer_text = response.content.strip()
            except Exception as exc:
                log.exception("LangChain generation failed during silent prime: %s", exc)
                return {
                    "primed": False,
                    "domain": domain,
                    "domain_info": context["domain_info"],
                    "used_image": context["used_image"],
                    "error": str(exc),
                }

            before_len = len(history_messages)
            state.memory.save_context({"question": question}, {"answer": answer_text})
            updated_history = state.memory.load_memory_variables({}).get("history", [])
            after_len = len(updated_history)
            if after_len > before_len:
                state.hidden_turns.extend(range(before_len, after_len))

            state.initial_summary = answer_text
            state.initial_summary_used = False

            return {
                "primed": True,
                "domain": domain,
                "domain_info": context["domain_info"],
                "used_image": context["used_image"],
            }

        if self._llm is None:
            log.warning("LLM client unavailable; delegating to qwen3vl fallback.")
            rag_answer = qwen3vl_generate(
                question,
                context["retrieved"],
                image_path=context.get("used_image"),
                domain=domain,
            )
            if rag_answer.get("success") and rag_answer.get("text"):
                answer_text = rag_answer["text"].strip()
            else:
                raw_reason = rag_answer.get("raw") or "Fallback generator returned no text"
                log.warning("qwen3vl fallback unavailable while LLM client disabled: %s", raw_reason)
                answer_text = "Generation failed. Retrieved context summary:\n" + context["context_block"]
            state.memory.save_context({"question": question}, {"answer": answer_text})
            return {
                "answer": answer_text,
                "domain": domain,
                "domain_info": context["domain_info"],
                "retrieved": context["retrieved"],
                "used_image": context["used_image"],
                "history": self.get_history(session_id),
                "memory_disabled": True,
            }

        try:
            response = self._llm.invoke(messages)
            answer_text = response.content.strip()
        except Exception as exc:
            log.exception("LangChain generation failed: %s", exc)
            answer_text = ""
            fallback_error: Optional[str] = None

            try:
                rag_response = qwen3vl_generate(
                    question,
                    context["retrieved"],
                    image_path=context.get("used_image"),
                    domain=domain,
                )
                if rag_response.get("success") and rag_response.get("text"):
                    answer_text = rag_response["text"].strip()
                else:
                    fallback_error = str(rag_response.get("raw") or "Fallback generator returned no text")
            except Exception as rag_exc:
                log.exception("Fallback qwen3vl generation failed: %s", rag_exc)
                fallback_error = str(rag_exc)

            if not answer_text:
                answer_text = "Generation failed. Retrieved context summary:\n" + context["context_block"]
                if fallback_error:
                    log.warning("Returning context summary due to generation failure: %s", fallback_error)

        state.memory.save_context({"question": question}, {"answer": answer_text})
        if state.initial_summary and not state.initial_summary_used and image_question:
            state.initial_summary_used = True

        if user_id:
            try:
                chat_repository.insert_message(
                    session_id=session_id,
                    role="user",
                    content=question,
                    attachment_url=image_path,
                    meta={"domain_hint": domain_hint} if domain_hint else None,
                )
                chat_repository.insert_message(
                    session_id=session_id,
                    role="assistant",
                    content=answer_text,
                    attachment_url=None,
                )
            except Exception as exc:
                log.exception("Failed to persist chat turn for %s: %s", session_id, exc)

        return {
            "answer": answer_text,
            "domain": domain,
            "domain_info": context["domain_info"],
            "retrieved": context["retrieved"],
            "used_image": context["used_image"],
            "history": self.get_history(session_id),
            "memory_disabled": self._llm is None,
        }


orchestrator = LangChainChatOrchestrator()

__all__ = ["LangChainChatOrchestrator", "orchestrator"]
