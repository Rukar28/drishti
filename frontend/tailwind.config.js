/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        vm: {
          bg: '#0B1020',
          surface: '#111827',
          'surface-2': '#172033',
          primary: '#35A9C8',
          secondary: '#4169B1',
          accent: '#8B82D9',
          text: '#F8FAFC',
          muted: '#94A3B8',
          success: '#22C55E',
          warning: '#F59E0B',
          danger: '#EF4444',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
