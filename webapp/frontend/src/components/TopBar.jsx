import { useTranslation } from "react-i18next";
import { Search } from "lucide-react";
import LanguageToggle from "./LanguageToggle";
import { useState } from "react";

export default function TopBar({ onSearch }) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) onSearch(query.trim());
  };

  return (
    <div className="absolute top-0 left-0 right-0 z-[1000] h-12 sm:h-14 bg-dark-800/90 backdrop-blur-md border-b border-dark-500/50 flex items-center px-2 sm:px-4 gap-2 sm:gap-4">
      {/* Logo — offset on mobile for hamburger */}
      <div className="flex items-center gap-2 flex-shrink-0 ml-10 sm:ml-0">
        <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-gradient-to-br from-fire-500 to-fire-700 flex items-center justify-center text-white font-bold text-xs sm:text-sm">
          🔥
        </div>
        <div className="hidden sm:block">
          <h1 className="text-sm font-bold text-white leading-none">FireSpreadNet</h1>
          <p className="text-[10px] text-gray-400 leading-none mt-0.5">{t("app_subtitle")}</p>
        </div>
      </div>

      {/* Search */}
      <form onSubmit={handleSubmit} className="flex-1 max-w-md mx-auto">
        <div className="relative">
          <Search className="absolute left-2 sm:left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-500" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("map.search_placeholder")}
            className="w-full pl-7 sm:pl-9 pr-3 sm:pr-4 py-1 sm:py-1.5 bg-dark-600 border border-dark-400 rounded-lg text-xs sm:text-sm text-white placeholder-gray-500 focus:outline-none focus:border-fire-500/50 focus:ring-1 focus:ring-fire-500/30 transition"
          />
        </div>
      </form>

      {/* Language toggle */}
      <LanguageToggle />
    </div>
  );
}
