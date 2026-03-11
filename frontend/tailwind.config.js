/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Primary: terracotta
        primary: {
          DEFAULT: '#FF7043',
          50: '#FFF3F0',
          100: '#FFE4DC',
          200: '#FFC9B8',
          300: '#FFAD95',
          400: '#FF8E6C',
          500: '#FF7043',
          600: '#F44D1A',
          700: '#C93200',
          800: '#962500',
          900: '#641900',
        },
        // Accent: amber
        accent: '#F59E0B',
        // Success: send color
        success: '#4CAF7D',
        // Background: chalk dust
        chalk: '#F7F5F2',
        // Gyms
        gym: {
          alpine: '#4CAF7D',
          mainwall: '#5B8DEE',
          progression: '#FF7043',
          all: '#F59E0B',
        },
        // Dark mode surfaces
        surface: {
          DEFAULT: '#F7F5F2',
          dark: '#1C1917',
        },
      },
      fontFamily: {
        sans: ['Nunito', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      maxWidth: {
        app: '440px',
      },
      aspectRatio: {
        reel: '9 / 16',
      },
    },
  },
  plugins: [],
}
