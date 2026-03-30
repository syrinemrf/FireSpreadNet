/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        fire: {
          50:  "#fff7ed",
          100: "#ffedd5",
          200: "#fed7aa",
          300: "#fdba74",
          400: "#fb923c",
          500: "#f97316",
          600: "#ea580c",
          700: "#c2410c",
          800: "#9a3412",
          900: "#7c2d12",
        },
        dark: {
          900: "#0a0e17",
          800: "#0f1520",
          700: "#151c2c",
          600: "#1c2538",
          500: "#243044",
          400: "#2d3b50",
        },
      },
    },
  },
  plugins: [],
}
