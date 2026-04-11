import type { Config } from 'tailwindcss'

export default {
  content: ['./src/**/*.{vue,ts,tsx,html}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Noto Sans Variable"', 'system-ui', 'sans-serif'],
        mono: ['"Noto Sans Mono Variable"', '"SF Mono"', 'Menlo', 'monospace'],
      },
      colors: {
        surface: {
          DEFAULT: '#ffffff',
          raised: '#f8f9fb',
          sunken: '#f1f3f5',
          border: '#e2e5e9',
          'border-subtle': '#eef0f3',
        },
        text: {
          DEFAULT: '#1a1d23',
          secondary: '#5c6370',
          muted: '#9ca3af',
        },
        accent: {
          DEFAULT: '#2563eb',
          hover: '#1d4ed8',
          subtle: '#eff6ff',
          border: '#bfdbfe',
        },
        status: {
          success: '#16a34a',
          'success-bg': '#f0fdf4',
          'success-border': '#bbf7d0',
          warning: '#ca8a04',
          'warning-bg': '#fefce8',
          'warning-border': '#fef08a',
          danger: '#dc2626',
          'danger-bg': '#fef2f2',
          'danger-border': '#fecaca',
          purple: '#7c3aed',
          'purple-bg': '#f5f3ff',
          'purple-border': '#ddd6fe',
          orange: '#ea580c',
          'orange-bg': '#fff7ed',
          'orange-border': '#fed7aa',
        },
      },
    },
  },
  plugins: [],
} satisfies Config
