import type { Config } from 'tailwindcss'

export default {
  content: ['./src/**/*.{vue,ts,html}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Noto Sans Variable"', 'system-ui', 'sans-serif'],
        mono: ['"Noto Sans Mono Variable"', 'monospace'],
      },
    },
  },
} satisfies Config
