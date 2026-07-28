import type { Config } from 'tailwindcss'

export default {
  content: ['./src/**/*.{vue,ts,html}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Noto Sans Variable"', 'system-ui', 'sans-serif'],
        mono: ['"Noto Sans Mono Variable"', 'monospace'],
      },
      colors: {
        ink: 'var(--color-ink)',
        moss: '#315c43',
        fern: '#477d5a',
        mist: '#edf2ec',
        cream: 'var(--color-cream)',
        line: '#d8ded6',
      },
      boxShadow: {
        card: '0 14px 45px rgba(35, 55, 41, 0.08)',
      },
    },
  },
  plugins: [],
} satisfies Config
