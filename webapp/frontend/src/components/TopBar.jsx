import { useTranslation } from "react-i18next";
import { Search, X, Menu } from "lucide-react";
import LanguageToggle from "./LanguageToggle";
import { useState } from "react";

export default function TopBar({ onSearch, onMenuToggle, sidebarOpen }) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) onSearch(query.trim());
  };

  const handleClear = () => setQuery("");

  return (
    <div className="absolute top-0 left-0 right-0 z-[1000] h-12 sm:h-14 bg-dark-900/95 backdrop-blur-md border-b border-dark-600/60 flex items-center px-2 sm:px-4 gap-2 sm:gap-3">

      {/* ── Hamburger (mobile only) ──────────── */}
      <button
        className="sm:hidden flex-shrink-0 w-9 h-9 rounded-lg bg-dark-700 border border-dark-500/60 flex items-center justify-center text-gray-300 hover:bg-dark-600 active:scale-95 transition"
        onClick={onMenuToggle}
        aria-label="Toggle menu"
      >
        {sidebarOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
      </button>

      {/* ── Brand (logo always visible) ───────── */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <img
          src="/logo.svg"
          alt="FireSpreadNet"
          className="w-7 h-7 sm:w-9 sm:h-9 flex-shrink-0"
          draggable={false}
        />
        <div className="leading-none hidden xs:block sm:block">
          <div className="text-xs sm:text-sm font-black text-white tracking-tight">
            <span className="text-fire-400">FIRE</span>SPREADNET
          </div>
          <div className="hidden sm:block text-[10px] text-gray-500 mt-0.5 font-medium">{t("app_subtitle")}</div>
        </div>
      </div>

      {/* ── Divider (desktop only) ─────────────── */}
      <div className="hidden sm:block w-px h-6 bg-dark-600 flex-shrink-0" />

      {/* ── Search bar ─────────────────────────── */}
      <form onSubmit={handleSubmit} className="flex-1 max-w-lg mx-auto">
        <div className={`relative flex items-center transition-all duration-200 ${focused ? "scale-[1.01]" : ""}`}>
          <Search className="absolute left-2.5 sm:left-3 w-3.5 h-3.5 text-gray-500 pointer-events-none flex-shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder={t("map.search_placeholder")}
            className={`
              w-full pl-8 sm:pl-9 pr-8 py-1.5 sm:py-2
              bg-dark-700/80 border rounded-xl text-xs sm:text-sm text-white
              placeholder-gray-500 transition-all duration-200
              focus:outline-none
              ${focused
                ? "border-fire-500/70 bg-dark-700 ring-2 ring-fire-500/20"
                : "border-dark-500/70 hover:border-dark-400"
              }
            `}
          />
          {query && (
            <button
              type="button"
              onClick={handleClear}
              className="absolute right-2.5 text-gray-500 hover:text-gray-300 transition"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </form>

      {/* ── Language toggle ─────────────────────── */}
      <div className="flex-shrink-0">
        <LanguageToggle />
      </div>
    </div>
  );
}


