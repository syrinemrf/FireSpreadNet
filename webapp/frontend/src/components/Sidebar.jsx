import { useTranslation } from "react-i18next";
import { Flame, Crosshair, Play, Brain, Info } from "lucide-react";

const items = [
  { key: "fires",       icon: Flame,     label: "sidebar.active_fires" },
  { key: "declare",     icon: Crosshair, label: "sidebar.declare_fire" },
  { key: "simulation",  icon: Play,      label: "sidebar.simulation" },
  { key: "explain",     icon: Brain,     label: "sidebar.explainability" },
  { key: "about",       icon: Info,       label: "sidebar.about" },
];

export default function Sidebar({
  activePanel,
  setActivePanel,
  declaring,
  setDeclaring,
  setShowAbout,
  onLoadFires,
  onExplain,
  fireCount,
  hasSimulation,
}) {
  const { t } = useTranslation();

  const handleClick = (key) => {
    if (key === "declare") {
      setDeclaring(!declaring);
      return;
    }
    if (key === "about") {
      setShowAbout(true);
      return;
    }
    if (key === "fires") {
      onLoadFires();
    }
    if (key === "explain" && onExplain) {
      onExplain();
      return;
    }
    setActivePanel(key);
  };

  return (
    <div className="w-16 h-full bg-dark-800 border-r border-dark-500/50 flex flex-col items-center py-4 sm:py-16 gap-1 z-[1001]">
      {items.map(({ key, icon: Icon, label }) => {
        const isActive = activePanel === key || (key === "declare" && declaring);
        return (
          <button
            key={key}
            onClick={() => handleClick(key)}
            title={t(label)}
            className={`
              relative w-12 h-12 rounded-xl flex flex-col items-center justify-center gap-0.5 transition-all
              ${isActive
                ? "bg-fire-600/20 text-fire-400 shadow-lg shadow-fire-600/10"
                : "text-gray-500 hover:bg-dark-600 hover:text-gray-300"
              }
            `}
          >
            <Icon className="w-5 h-5" />
            <span className="text-[9px] leading-none">{t(label).split(" ")[0]}</span>
            {key === "fires" && fireCount > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-fire-600 rounded-full text-[8px] flex items-center justify-center text-white font-bold">
                {fireCount > 99 ? "99+" : fireCount}
              </span>
            )}
            {key === "simulation" && hasSimulation && (
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse" />
            )}
          </button>
        );
      })}
    </div>
  );
}
