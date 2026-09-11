import type { Config } from 'tailwindcss'

export default {
  content: ['./src/**/*.{vue,ts,tsx,html}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Noto Sans Variable"', 'system-ui', 'sans-serif'],
        mono: ['"Noto Sans Mono Variable"', '"SF Mono"', 'Menlo', 'monospace'],
      },
      colors: {
        surface: {
          DEFAULT: 'var(--c-surface)',
          raised: 'var(--c-surface-raised)',
          sunken: 'var(--c-surface-sunken)',
          border: 'var(--c-surface-border)',
          'border-subtle': 'var(--c-surface-border-subtle)',
        },
        text: {
          DEFAULT: 'var(--c-text)',
          secondary: 'var(--c-text-secondary)',
          muted: 'var(--c-text-muted)',
        },
        accent: {
          DEFAULT: 'var(--c-accent)',
          hover: 'var(--c-accent-hover)',
          subtle: 'var(--c-accent-subtle)',
          border: 'var(--c-accent-border)',
        },
        status: {
          success: 'var(--c-status-success)',
          'success-bg': 'var(--c-status-success-bg)',
          'success-border': 'var(--c-status-success-border)',
          warning: 'var(--c-status-warning)',
          'warning-bg': 'var(--c-status-warning-bg)',
          'warning-border': 'var(--c-status-warning-border)',
          danger: 'var(--c-status-danger)',
          'danger-bg': 'var(--c-status-danger-bg)',
          'danger-border': 'var(--c-status-danger-border)',
        },
      },
    },
  },
  plugins: [],
} satisfies Config
