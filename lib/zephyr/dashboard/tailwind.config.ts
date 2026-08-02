import type { Config } from 'tailwindcss'

export default {
  content: ['./src/**/*.{vue,ts,html}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"SFMono-Regular"', 'Consolas', 'monospace'],
      },
      colors: {
        surface: {
          DEFAULT: 'var(--c-surface)',
          raised: 'var(--c-surface-raised)',
          sunken: 'var(--c-surface-sunken)',
          border: 'var(--c-border)',
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
        },
        status: {
          success: 'var(--c-success)',
          'success-bg': 'var(--c-success-bg)',
          warning: 'var(--c-warning)',
          'warning-bg': 'var(--c-warning-bg)',
          danger: 'var(--c-danger)',
          'danger-bg': 'var(--c-danger-bg)',
        },
      },
      boxShadow: {
        card: '0 1px 2px rgb(15 23 42 / 0.05), 0 4px 16px rgb(15 23 42 / 0.04)',
      },
    },
  },
  plugins: [],
} satisfies Config
