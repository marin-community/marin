import type { Config } from 'tailwindcss'

// Colors resolve to CSS variables defined in src/styles/main.css (light + .dark).
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
          success: 'var(--c-status-success)',
          warning: 'var(--c-status-warning)',
          danger: 'var(--c-status-danger)',
        },
      },
    },
  },
  plugins: [],
} satisfies Config
