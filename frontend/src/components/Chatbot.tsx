import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  MessageCircle,
  Send,
  Bot,
  User,
  Loader2,
  Image as ImageIcon,
  Plus,
  Trash2,
  Paperclip,
  X,
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { apiFetch, API_BASE_URL } from '../lib/api';

interface ChatbotProps {
  onSessionChange?: (sessionId: string) => void;
}

interface ChatHistoryMessage {
  type: 'human' | 'ai' | string;
  content: string;
}

interface ChatAttachment {
  url: string;
  name: string;
  preview?: string;
  originalSize?: number;
  compressedSize?: number;
}

interface RenderMessage extends ChatHistoryMessage {
  attachments?: ChatAttachment[];
}

interface PreparedAttachment {
  file: File;
  previewUrl: string;
  name: string;
  originalSize: number;
  compressedSize: number;
}

const MAX_SESSION_TITLE_LENGTH = 60;
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const isValidUuid = (value: unknown): value is string => typeof value === 'string' && UUID_REGEX.test(value ?? '');

const truncate = (value: string, limit: number) => {
  if (value.length <= limit) return value;
  return `${value.slice(0, limit - 1)}…`;
};

async function compressImage(file: File): Promise<{ file: File; originalSize: number; compressedSize: number }> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error ?? new Error('Unable to read image'));
    reader.readAsDataURL(file);
  });

  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Unable to load image for compression'));
    img.src = dataUrl;
  });

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas context unavailable for compression');
  }

  const maxDimension = 1600;
  let { width, height } = image;
  if (width > height && width > maxDimension) {
    const ratio = maxDimension / width;
    width = maxDimension;
    height = Math.round(height * ratio);
  } else if (height > maxDimension) {
    const ratio = maxDimension / height;
    height = maxDimension;
    width = Math.round(width * ratio);
  }

  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(image, 0, 0, width, height);

  const compressedBlob: Blob = await new Promise((resolve, reject) => {
    canvas.toBlob(
      blob => (blob ? resolve(blob) : reject(new Error('Failed to generate compressed image'))),
      'image/jpeg',
      0.85,
    );
  });

  const baseName = file.name.replace(/\.[^/.]+$/, '') || 'image';
  const compressedFile = new File([compressedBlob], `${baseName}-optimized.jpg`, { type: 'image/jpeg' });

  return {
    file: compressedFile,
    originalSize: file.size,
    compressedSize: compressedFile.size,
  };
}

async function uploadImage(file: File): Promise<{ url: string }> {
  const formData = new FormData();
  const originalName = file.name;
  const uploadName = originalName.includes('.') ? originalName : `${originalName}.jpg`;
  formData.append('file', file, uploadName);

  const response = await fetch(`${API_BASE_URL}/upload-image`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to upload image');
  }

  const data = await response.json();
  if (!data || data.status !== 'success' || !data.file_url) {
    throw new Error(data?.message ?? 'Unexpected response from upload endpoint');
  }

  return { url: data.file_url as string };
}
interface ChatResponse {
  success: boolean;
  answer: string;
  domain: string;
  domain_info: Record<string, any>;
  retrieved: Array<Record<string, any>>;
  used_image: string | null;
  history: ChatHistoryMessage[];
  memory_disabled?: boolean;
}

interface ChatSessionMeta {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  excerpt?: string;
}

interface SessionContextSnapshot {
  retrieved: ChatResponse['retrieved'];
  domainInfo: Record<string, any> | null;
  memoryDisabled: boolean;
  attachment: ChatAttachment | null;
}

export default function Chatbot({ onSessionChange }: ChatbotProps) {
  const { user } = useAuth();
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<RenderMessage[]>([]);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [retrievedContext, setRetrievedContext] = useState<ChatResponse['retrieved']>([]);
  const [domainInfo, setDomainInfo] = useState<Record<string, any> | null>(null);
  const [memoryDisabled, setMemoryDisabled] = useState(false);
  const [sessions, setSessions] = useState<ChatSessionMeta[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [pendingAttachments, setPendingAttachments] = useState<PreparedAttachment[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [sessionAttachments, setSessionAttachments] = useState<Record<string, ChatAttachment | null>>({});
  const [historyCache, setHistoryCache] = useState<Record<string, RenderMessage[]>>({});
  const [contextCache, setContextCache] = useState<Record<string, SessionContextSnapshot>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const storageKey = useMemo(() => {
    if (!user) return null;
    return `medrag-chat-sessions-${user.id}`;
  }, [user]);

  const persistSessions = useCallback(
    (next: ChatSessionMeta[]) => {
      if (!storageKey) return;
      try {
        localStorage.setItem(storageKey, JSON.stringify(next));
      } catch (error) {
        console.warn('Unable to persist chat sessions to localStorage:', error);
      }
    },
    [storageKey]
  );

  const loadSessions = useCallback(() => {
    if (!storageKey) {
      setSessions([]);
      setActiveSessionId(null);
      return;
    }

    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) {
        setSessions([]);
        setActiveSessionId(null);
        return;
      }
      const parsed = JSON.parse(raw) as ChatSessionMeta[];
      const validSessions = Array.isArray(parsed)
        ? parsed.filter(session => session && isValidUuid(session.id))
        : [];
      if (validSessions.length > 0) {
        const ordered = validSessions.sort((a, b) => b.updatedAt - a.updatedAt);
        setSessions(ordered);
        setActiveSessionId(ordered[0].id);
      } else {
        setSessions([]);
        setActiveSessionId(null);
      }
    } catch (error) {
      console.warn('Failed to parse stored chat sessions:', error);
      setSessions([]);
      setActiveSessionId(null);
    }
  }, [storageKey]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    if (sessions.length === 0 && user && !loading) {
      const newSession: ChatSessionMeta = {
        id: crypto.randomUUID(),
        title: 'New chat',
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      setSessions([newSession]);
      setActiveSessionId(newSession.id);
      persistSessions([newSession]);
    }
  }, [sessions.length, persistSessions, user, loading]);

  useEffect(() => {
    if (!sessions.length || !storageKey) return;
    persistSessions(sessions);
  }, [sessions, storageKey, persistSessions]);

  useEffect(() => {
    if (activeSessionId) {
      onSessionChange?.(activeSessionId);
    }
  }, [activeSessionId, onSessionChange]);

  useEffect(() => {
    if (!activeSessionId) return;

    const cachedHistory = historyCache[activeSessionId];
    if (cachedHistory) {
      setHistory(cachedHistory);
      const cachedContext = contextCache[activeSessionId];
      setRetrievedContext(cachedContext?.retrieved ?? []);
      setDomainInfo(cachedContext?.domainInfo ?? null);
      setMemoryDisabled(Boolean(cachedContext?.memoryDisabled));
      return;
    }

    const fetchHistory = async () => {
      try {
        const data = await apiFetch<{ success: boolean; history: ChatHistoryMessage[] }>(
          `/chat/history/${activeSessionId}`
        );
        const fetchedHistory = data.history ?? [];
        const mappedHistory = fetchedHistory.map(msg => ({ ...msg })) as RenderMessage[];
        setHistory(mappedHistory);
        setHistoryCache(prev => ({ ...prev, [activeSessionId]: mappedHistory }));
        setRetrievedContext([]);
        setDomainInfo(null);
        setMemoryDisabled(false);
        setContextCache(prev => ({
          ...prev,
          [activeSessionId]: prev[activeSessionId] ?? {
            retrieved: [],
            domainInfo: null,
            memoryDisabled: false,
            attachment: sessionAttachments[activeSessionId] ?? null,
          },
        }));
      } catch (error) {
        console.error('Failed to fetch chat history:', error);
      }
    };
    fetchHistory();
  }, [activeSessionId, historyCache, contextCache, sessionAttachments]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, pendingAttachments]);

  const updateSessionMeta = useCallback(
    (sessionId: string, updates: Partial<ChatSessionMeta>) => {
      setSessions(prev => {
        const updated = prev.map(session =>
          session.id === sessionId
            ? { ...session, ...updates, updatedAt: updates.updatedAt ?? Date.now() }
            : session
        );
        return updated.sort((a, b) => b.updatedAt - a.updatedAt);
      });
    },
    []
  );

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if ((!input.trim() && pendingAttachments.length === 0) || !activeSessionId || loading) {
      return;
    }

    setLoading(true);
    setSessionError(null);

    try {
      let uploadedAttachment: ChatAttachment | null = null;
      if (pendingAttachments.length > 0) {
        const attachment = pendingAttachments[0];
        const { file, previewUrl, name, originalSize, compressedSize } = attachment;
        const { url } = await uploadImage(file);
        uploadedAttachment = {
          url,
          name,
          preview: url,
          originalSize,
          compressedSize,
        };
        if (previewUrl.startsWith('blob:')) {
          URL.revokeObjectURL(previewUrl);
        }
      }

      const payload: Record<string, unknown> = {
        session_id: activeSessionId,
        question: input.trim(),
        image_path: uploadedAttachment?.url ?? null,
        domain: null,
        user_id: user?.id ?? null,
      };

      const response = await apiFetch<ChatResponse>('/chat/message', {
        method: 'POST',
        body: JSON.stringify(payload),
      });

      if (!response.success) {
        throw new Error('Chat endpoint returned unsuccessful response.');
      }

      const nextHistory = (response.history ?? []).map(msg => ({ ...msg })) as RenderMessage[];
      if (uploadedAttachment) {
        const lastHumanIndex = [...nextHistory].reverse().findIndex(msg => msg.type === 'human');
        if (lastHumanIndex !== -1) {
          const indexFromStart = nextHistory.length - 1 - lastHumanIndex;
          nextHistory[indexFromStart] = {
            ...nextHistory[indexFromStart],
            attachments: [...(nextHistory[indexFromStart].attachments ?? []), uploadedAttachment],
          };
        }
      }

      setHistory(nextHistory);
      setRetrievedContext(response.retrieved ?? []);
      setDomainInfo(response.domain_info ?? null);
      setMemoryDisabled(Boolean(response.memory_disabled));
      setInput('');
      setPendingAttachments([]);
      setInfoMessage(null);

      setHistoryCache(prev => ({ ...prev, [activeSessionId]: nextHistory }));
      setContextCache(prev => ({
        ...prev,
        [activeSessionId]: {
          retrieved: response.retrieved ?? [],
          domainInfo: response.domain_info ?? null,
          memoryDisabled: Boolean(response.memory_disabled),
          attachment: uploadedAttachment ?? prev[activeSessionId]?.attachment ?? sessionAttachments[activeSessionId] ?? null,
        },
      }));
      if (uploadedAttachment) {
        setSessionAttachments(prev => ({
          ...prev,
          [activeSessionId]: uploadedAttachment,
        }));
      }

      if (response.history && response.history.length > 0) {
        const firstHuman = response.history.find(msg => msg.type === 'human');
        if (firstHuman) {
          const title = truncate(firstHuman.content.replace(/\s+/g, ' ').trim(), MAX_SESSION_TITLE_LENGTH);
          updateSessionMeta(activeSessionId, {
            title: title || 'New chat',
            excerpt: truncate(response.answer ?? '', 80),
          });
        } else {
          updateSessionMeta(activeSessionId, {
            excerpt: truncate(response.answer ?? '', 80),
          });
        }
      }
    } catch (error: any) {
      console.error('Failed to send chat message:', error);
      setSessionError(error.message ?? 'Failed to contact chat server.');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectSession = (sessionId: string) => {
    if (loading) return;
    setActiveSessionId(sessionId);
    setSessionError(null);
    setInfoMessage(null);
    clearPendingAttachments();
    setDragActive(false);
  };

  const handleNewSession = () => {
    if (!user) return;
    const newSessionId = crypto.randomUUID();
    const newSession: ChatSessionMeta = {
      id: newSessionId,
      title: 'New chat',
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setSessions(prev => [newSession, ...prev]);
    setActiveSessionId(newSessionId);
    setHistory([]);
    setRetrievedContext([]);
    setDomainInfo(null);
    setSessionError(null);
    setInfoMessage(null);
    setMemoryDisabled(false);
    setPendingAttachments(prev => {
      prev.forEach(item => {
        if (item.previewUrl.startsWith('blob:')) {
          URL.revokeObjectURL(item.previewUrl);
        }
      });
      return [];
    });
    setDragActive(false);
    setHistoryCache(prev => ({ ...prev, [newSessionId]: [] }));
    setContextCache(prev => ({
      ...prev,
      [newSessionId]: {
        retrieved: [],
        domainInfo: null,
        memoryDisabled: false,
        attachment: null,
      },
    }));

    onSessionChange?.(newSessionId);
  };

  const activeDomainInfo = domainInfo;
  const shouldShowRetrievalDetails = useMemo(() => {
    if (!activeDomainInfo) {
      return false;
    }
    const domainName = String(activeDomainInfo.domain ?? '').toLowerCase();
    const allowedDomains = new Set(['ophthalmology', 'radiology']);
    if (!domainName || !allowedDomains.has(domainName)) {
      return false;
    }
    return retrievedContext.length > 0 || Object.keys(activeDomainInfo).length > 0;
  }, [activeDomainInfo, retrievedContext.length]);

  const clearPendingAttachments = useCallback(() => {
    setPendingAttachments(prev => {
      prev.forEach(item => {
        if (item.previewUrl.startsWith('blob:')) {
          URL.revokeObjectURL(item.previewUrl);
        }
      });
      return [];
    });
  }, []);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const { file: compressed, originalSize, compressedSize } = await compressImage(file);
      const previewUrl = URL.createObjectURL(compressed);

      setPendingAttachments(prev => {
        prev.forEach(item => URL.revokeObjectURL(item.previewUrl));
        return [
          {
            file: compressed,
            previewUrl,
            name: compressed.name,
            originalSize,
            compressedSize,
          },
        ];
      });
      setSessionError(null);
    } catch (error: any) {
      console.error('Failed to prepare attachment:', error);
      setSessionError(error.message ?? 'Failed to prepare image');
    } finally {
      if (event.target) {
        event.target.value = '';
      }
    }
  };

  const handleClearAttachment = () => {
    clearPendingAttachments();
  };

  const handleDragOver = (event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (!dragActive) {
      setDragActive(true);
    }
  };

  const handleDragLeave = (event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const relatedTarget = event.relatedTarget as Node | null;
    if (relatedTarget && event.currentTarget.contains(relatedTarget)) {
      return;
    }
    if (dragActive) {
      setDragActive(false);
    }
  };

  const handleDrop = async (event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);

    const file = event.dataTransfer.files?.[0];
    if (!file) return;

    try {
      const { file: compressed, originalSize, compressedSize } = await compressImage(file);
      const previewUrl = URL.createObjectURL(compressed);

      setPendingAttachments(prev => {
        prev.forEach(item => URL.revokeObjectURL(item.previewUrl));
        return [
          {
            file: compressed,
            previewUrl,
            name: compressed.name,
            originalSize,
            compressedSize,
          },
        ];
      });
    } catch (error: any) {
      console.error('Failed to prepare dropped attachment:', error);
      setSessionError(error.message ?? 'Failed to prepare dropped image');
    }
  };

  const handleDeleteSession = (sessionId: string) => {
    if (!window.confirm('Delete this conversation? This cannot be undone.')) {
      return;
    }

    setSessions(prev => prev.filter(session => session.id !== sessionId));
    setHistoryCache(prev => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
    setContextCache(prev => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });

    if (sessionId === activeSessionId) {
      const remainingSessions = sessions.filter(session => session.id !== sessionId).sort((a, b) => b.updatedAt - a.updatedAt);
      if (remainingSessions.length > 0) {
        setActiveSessionId(remainingSessions[0].id);
      } else {
        setActiveSessionId(null);
        setHistory([]);
        setRetrievedContext([]);
        setDomainInfo(null);
        clearPendingAttachments();
        setSessionError(null);
        setInfoMessage(null);
        setDragActive(false);
      }
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg flex h-[calc(100vh-140px)]">
      <aside className="w-64 border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <div className="bg-green-100 p-2 rounded-lg">
              <MessageCircle className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-800">Conversations</h2>
              <p className="text-xs text-gray-500">Switch between past chats</p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleNewSession}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium py-2.5 rounded-lg transition-colors disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-2">
          {sessions.length === 0 ? (
            <div className="text-sm text-gray-500 px-2 py-4">No chats yet. Start a conversation.</div>
          ) : (
            sessions.map(session => {
              const isActive = session.id === activeSessionId;
              return (
                <div
                  key={session.id}
                  className={`w-full px-3 py-3 rounded-lg transition-colors border ${isActive ? 'bg-blue-50 border-blue-200 text-blue-700' : 'bg-white border-transparent hover:border-blue-200 hover:bg-blue-50'
                    }`}
                >
                  <button
                    type="button"
                    onClick={() => handleSelectSession(session.id)}
                    className="w-full text-left"
                  >
                    <p className="text-sm font-semibold truncate">{session.title || 'New chat'}</p>
                    <p className="text-xs text-gray-500 truncate">{session.excerpt || 'Draft a question to begin.'}</p>
                  </button>
                  <div className="flex items-center justify-between mt-3">
                    <span className="text-[11px] text-gray-400">
                      {new Date(session.updatedAt).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                    </span>
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        handleDeleteSession(session.id);
                      }}
                      className="flex items-center justify-center gap-1 text-xs px-2 py-1 rounded-md border border-red-300 text-red-600 hover:bg-red-50 transition-colors"
                    >
                      <Trash2 className="w-3 h-3" />
                      Delete
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </aside>

      <div
        className={`flex-1 flex flex-col transition-colors ${dragActive ? 'bg-blue-50/60' : ''}`}
        onDragEnter={handleDragOver}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex items-center justify-between gap-3 p-6 border-b border-gray-200">
          <div>
            <h3 className="text-2xl font-bold text-gray-800">Medical Assistant</h3>
            <p className="text-sm text-gray-600">Ask questions and attach reference images</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => activeSessionId && handleDeleteSession(activeSessionId)}
              className="flex items-center gap-2 px-3 py-2 border border-red-300 text-red-600 rounded-lg hover:bg-red-50 transition-colors"
              disabled={!activeSessionId || loading}
            >
              <Trash2 className="w-4 h-4" />
              Delete Chat
            </button>
          </div>
        </div>

        <div
          className={`flex-1 overflow-y-auto p-6 space-y-4 transition-colors ${dragActive ? 'bg-blue-50 border-blue-200 border-dashed border-2 rounded-xl' : ''}`}
        >
          {dragActive && (
            <div className="flex flex-col items-center justify-center h-40 border border-blue-300 border-dashed rounded-xl bg-white/60">
              <ImageIcon className="w-10 h-10 text-blue-500 mb-2" />
              <p className="text-sm font-semibold text-blue-700">Drop image anywhere to attach</p>
            </div>
          )}

          {memoryDisabled && (
            <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded-lg text-sm">
              LLM key not detected. Responses include retrieved context summaries only.
            </div>
          )}

          {sessionError && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
              {sessionError}
            </div>
          )}

          {infoMessage && (
            <div className="bg-blue-50 border border-blue-200 text-blue-700 px-4 py-3 rounded-lg text-sm">
              {infoMessage}
            </div>
          )}

          {!activeSessionId ? (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-500">
              <Bot className="w-16 h-16 text-gray-300 mb-4" />
              <p className="mb-2">No active conversation selected</p>
              <p className="text-sm text-gray-400">Choose an existing chat or start a new one.</p>
            </div>
          ) : history.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-500">
              <Bot className="w-16 h-16 text-gray-300 mb-4" />
              <p className="mb-2">No conversation yet</p>
              <p className="text-sm text-gray-400">Ask a question to start building session memory.</p>
            </div>
          ) : (
            <>
              <div className="space-y-4">
                {history.map((msg, index) => {
                  const isHuman = msg.type === 'human';
                  const iconWrapper = isHuman ? (
                    <div className="bg-blue-100 p-2 rounded-lg flex-shrink-0">
                      <User className="w-5 h-5 text-blue-600" />
                    </div>
                  ) : (
                    <div className="bg-green-100 p-2 rounded-lg flex-shrink-0">
                      <Bot className="w-5 h-5 text-green-600" />
                    </div>
                  );

                  const bubbleClass = isHuman ? 'bg-blue-50' : 'bg-gray-50';

                  return (
                    <div key={`${activeSessionId}-${index}`} className="space-y-2">
                      <div className="flex items-start gap-3">
                        {iconWrapper}
                        <div className={`${bubbleClass} rounded-lg p-3 flex-1 space-y-3`}>
                          <p className="text-gray-800 whitespace-pre-line">{msg.content}</p>
                          {msg.attachments && msg.attachments.length > 0 && (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                              {msg.attachments.map(attachment => (
                                <figure key={attachment.url} className="bg-white rounded-lg overflow-hidden shadow-sm border border-gray-200">
                                  <img src={attachment.preview ?? attachment.url} alt={attachment.name} className="w-full h-40 object-cover" />
                                  <figcaption className="p-2 text-xs text-gray-600 truncate">
                                    {attachment.name}
                                    {attachment.compressedSize && attachment.originalSize && (
                                      <span className="block text-[10px] text-gray-400">
                                        {Math.round(attachment.compressedSize / 1024)} KB (was {Math.round(attachment.originalSize / 1024)} KB)
                                      </span>
                                    )}
                                  </figcaption>
                                </figure>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {shouldShowRetrievalDetails && (
                <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-3 text-sm text-gray-700">
                  <h3 className="text-sm font-semibold text-gray-800">Latest retrieval details</h3>
                  {retrievedContext.length > 0 && (
                    <details className="space-y-2">
                      <summary className="cursor-pointer text-blue-600">View retrieved cases</summary>
                      <ul className="list-disc ml-5 space-y-2">
                        {retrievedContext.map((item, idx) => (
                          <li key={idx}>
                            <p className="font-medium text-gray-700">{item.image_path ?? `Case ${idx + 1}`}</p>
                            <p className="text-gray-600 whitespace-pre-line">{item.report_text}</p>
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                  {activeDomainInfo && (
                    <p className="text-xs text-gray-500">
                      Domain inferred: {String(activeDomainInfo.domain ?? 'Unknown')} (method: {String(activeDomainInfo.method ?? 'n/a')}, score: {typeof activeDomainInfo.score === 'number' ? activeDomainInfo.score.toFixed(2) : 'n/a'})
                    </p>
                  )}
                </div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        <div className="border-t border-gray-200 p-4 space-y-4">
          {pendingAttachments.length > 0 && (
            <div className="border border-blue-200 rounded-lg p-3 bg-blue-50/60">
              {pendingAttachments.map(attachment => (
                <div key={attachment.previewUrl} className="flex items-center gap-3">
                  <img src={attachment.previewUrl} alt={attachment.name} className="w-16 h-16 object-cover rounded" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-700">{attachment.name}</p>
                    <p className="text-xs text-gray-500">
                      {Math.round(attachment.compressedSize / 1024)} KB (was {Math.round(attachment.originalSize / 1024)} KB)
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleClearAttachment}
                    className="text-red-500 hover:text-red-600"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <form
            onSubmit={handleSubmit}
            onDragEnter={handleDragOver}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`flex items-center gap-3 transition-colors ${dragActive ? 'bg-blue-50/60 border border-blue-200 rounded-lg p-2' : ''}`}
          >
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              onDragEnter={handleDragOver}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className="p-3 border border-gray-300 rounded-lg text-gray-600 hover:text-blue-600 hover:border-blue-500 transition-colors disabled:opacity-50"
              disabled={loading}
            >
              <Paperclip className="w-5 h-5" />
            </button>
            <input
              type="text"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask a question about medical imaging..."
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              disabled={loading || !activeSessionId}
            />
            <button
              type="submit"
              disabled={loading || (!input.trim() && pendingAttachments.length === 0) || !activeSessionId}
              className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading ? <Loader2 className="w-6 h-6 animate-spin" /> : <Send className="w-6 h-6" />}
            </button>
          </form>
          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
        </div>
      </div>
    </div>
  );
}
