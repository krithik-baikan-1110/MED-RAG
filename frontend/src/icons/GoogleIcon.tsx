import type { SVGProps } from 'react';

export default function GoogleIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
      role="img"
      {...props}
    >
      <path
        fill="#EA4335"
        d="M12 11v3.7h5.3c-.2 1.4-.9 2.6-2.1 3.4l3.4 2.6c2-1.8 3.1-4.4 3.1-7.5 0-.7-.1-1.4-.2-2H12z"
      />
      <path
        fill="#34A853"
        d="M5.3 14.3l-.8.6-2.7 2.1C3.4 20.3 7.4 23 12 23c2.7 0 5-.9 6.6-2.3l-3.4-2.6c-.9.6-2 1-3.2 1-2.5 0-4.6-1.7-5.4-3.9l-.2-.9z"
      />
      <path
        fill="#4A90E2"
        d="M2.6 6.9C1.9 8.3 1.5 9.9 1.5 11.5s.4 3.2 1.1 4.6c0 .1 2.7-2.1 2.7-2.1-.2-.6-.4-1.3-.4-2s.1-1.4.4-2l-2.7-2.1z"
      />
      <path
        fill="#FBBC05"
        d="M12 4.8c1.5 0 2.8.5 3.9 1.4l2.9-2.9C16.9 1.3 14.7.5 12 .5 7.4.5 3.4 3.2 2.1 6.9l2.7 2.1C5.6 6.8 7.7 4.8 12 4.8z"
      />
    </svg>
  );
}
