/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        dawson: '#e74c3c',
        denver: '#3498db',
      },
    },
  },
  plugins: [],
}
