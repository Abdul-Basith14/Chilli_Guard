/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Near-black backgrounds
        surface: {
          950: '#0a0a0b',
          900: '#111113',
          800: '#18181c',
          700: '#1f1f24',
          600: '#26262d',
        },
        // Olive-green accent palette
        olive: {
          50:  '#f6f7e8',
          100: '#eaedd1',
          200: '#d4dba3',
          300: '#b9c46d',
          400: '#9aaa3e',
          500: '#7d8f28',
          600: '#62711e',
          700: '#4c561a',
          800: '#3e4519',
          900: '#363c19',
          950: '#1c200a',
        },
        // Off-white text
        ink: {
          50:  '#f8f8f6',
          100: '#f0f0ec',
          200: '#e0e0d8',
          300: '#c8c8bc',
          400: '#a8a898',
          500: '#888876',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'olive-glow': 'radial-gradient(ellipse at 50% 0%, rgba(125,143,40,0.15) 0%, transparent 60%)',
      },
      boxShadow: {
        'olive-sm': '0 0 12px rgba(125,143,40,0.15)',
        'olive-md': '0 0 24px rgba(125,143,40,0.2)',
        'olive-lg': '0 0 48px rgba(125,143,40,0.25)',
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease-in-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'pulse-olive': 'pulseOlive 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 3s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseOlive: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
      },
    },
  },
  plugins: [],
}
