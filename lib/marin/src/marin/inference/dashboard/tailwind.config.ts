import type { Config } from 'tailwindcss'

// Colors resolve to CSS variables defined in src/styles/main.css (light + .dark).
// System font stacks keep the single-file build free of font assets.
export default {
  content: ['./src/**/*.{vue,ts,tsx,html}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['system-ui', '-apple-system', '"Segoe UI"', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', '"SF Mono"', 'Menlo', 'Consolas', 'monospace'],
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
          danger: 'var(--c-status-danger)',
        },
      },
    },
  },
  plugins: [],
} satisfies Config
