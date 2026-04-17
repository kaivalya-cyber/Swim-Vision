/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    container: {
      center: true,
      padding: "1rem",
      screens: {
        "2xl": "1280px",
      },
    },
    extend: {
      colors: {
        border: "hsl(0 0% 100% / 0.09)",
        input: "hsl(0 0% 100% / 0.09)",
        ring: "hsl(0 0% 100% / 0.16)",
        background: "hsl(0 0% 2%)",
        foreground: "hsl(0 0% 96%)",
        primary: {
          DEFAULT: "hsl(0 0% 98%)",
          foreground: "hsl(0 0% 7%)",
        },
        secondary: {
          DEFAULT: "hsl(0 0% 100% / 0.08)",
          foreground: "hsl(0 0% 96%)",
        },
        muted: {
          DEFAULT: "hsl(0 0% 100% / 0.08)",
          foreground: "hsl(0 0% 72%)",
        },
        accent: {
          DEFAULT: "hsl(0 0% 100% / 0.1)",
          foreground: "hsl(0 0% 96%)",
        },
        destructive: {
          DEFAULT: "hsl(0 84% 60%)",
          foreground: "hsl(0 0% 98%)",
        },
        card: {
          DEFAULT: "hsl(0 0% 100% / 0.04)",
          foreground: "hsl(0 0% 96%)",
        },
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.5rem",
        "3xl": "2rem",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.06), 0 30px 80px rgba(255,255,255,0.08)",
        soft: "0 25px 70px rgba(0,0,0,0.35)",
      },
      fontFamily: {
        sans: ["Inter", "Avenir Next", "Segoe UI", "sans-serif"],
      },
      backgroundImage: {
        grain:
          "radial-gradient(circle at 20% 20%, rgba(255,255,255,0.09), transparent 26%), radial-gradient(circle at 80% 30%, rgba(255,255,255,0.06), transparent 22%), linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0))",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translate3d(0, 0, 0) scale(1)" },
          "50%": { transform: "translate3d(0, -12px, 0) scale(1.03)" },
        },
        shimmer: {
          "0%": { opacity: "0.4", transform: "translateX(-8%)" },
          "50%": { opacity: "0.85" },
          "100%": { opacity: "0.4", transform: "translateX(8%)" },
        },
      },
      animation: {
        float: "float 12s ease-in-out infinite",
        shimmer: "shimmer 10s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
