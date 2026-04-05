import { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Flame, Crosshair, Play, Brain, Info,
  ShieldAlert, ChevronLeft, ChevronRight,
  Activity, Satellite, X,
} from "lucide-react";

/* ── Navigation structure ─────────────────────────────────────── */
const SECTIONS = [
  {
    group: "sidebar.group_monitor",
    items: [
      {
        key: "fires",
        icon: Satellite,
        label: "sidebar.active_fires",
        color: { active: "bg-red-600/20 text-red-300 border-red-600/30", ring: "ring-red-500/40", dot: "bg-red-500" },
      },
      {
        key: "risk",
        icon: ShieldAlert,
        label: "sidebar.risk",
        color: { active: "bg-amber-600/20 text-amber-300 border-amber-600/30", ring: "ring-amber-500/40", dot: "bg-amber-400" },
      },
    ],
  },
  {
    group: "sidebar.group_respond",
    items: [
      {
        key: "declare",
        icon: Crosshair,
        label: "sidebar.declare_fire",
        color: { active: "bg-blue-600/20 text-blue-300 border-blue-600/30", ring: "ring-blue-500/40", dot: "bg-blue-400" },
      },
      {
        key: "simulation",
        icon: Play,
        label: "sidebar.simulation",
        color: { active: "bg-green-600/20 text-green-300 border-green-600/30", ring: "ring-green-500/40", dot: "bg-green-400" },
      },
    ],
  },
  {
    group: "sidebar.group_analyze",
    items: [
      {
        key: "explain",
        icon: Brain,
        label: "sidebar.explainability",
        color: { active: "bg-purple-600/20 text-purple-300 border-purple-600/30", ring: "ring-purple-500/40", dot: "bg-purple-400" },
      },
    ],
  },
  {
    group: "sidebar.group_info",
    items: [
      {
        key: "about",
        icon: Info,
        label: "sidebar.about",
        color: { active: "bg-dark-500/60 text-gray-300 border-dark-400/40", ring: "ring-gray-500/40", dot: "bg-gray-400" },
      },
    ],
  },
];

/* ── Main component ───────────────────────────────────────────── */
export default function Sidebar({
  activePanel,
  setActivePanel,
  declaring,
  setDeclaring,
  simulating,
  setSimulating,
  setShowAbout,
  onLoadFires,
  onExplain,
  onRisk,
  onClearRisk,
  fireCount,
  hasSimulation,
  hasRisk,
  riskLoading,
  pickingRisk,
  onMobileClose,
}) {
  const { t } = useTranslation();

  /* Desktop collapsed/expanded state (mobile always expanded in overlay) */
  const [expanded, setExpanded] = useState(
    () => localStorage.getItem("fsn_sidebar") !== "collapsed"
  );

  const toggleExpanded = () => {
    const next = !expanded;
    setExpanded(next);
    localStorage.setItem("fsn_sidebar", next ? "expanded" : "collapsed");
  };

  /* Click handling ─────────────────────────────── */
  const handleClick = (key) => {
    if (key === "declare") {
      setDeclaring(!declaring);
      if (simulating) setSimulating(false);
    } else if (key === "simulation") {
      setSimulating(!simulating);
      if (declaring) setDeclaring(false);
    } else if (key === "risk") {
      if (hasRisk) onClearRisk();
      else if (onRisk) onRisk();
    } else if (key === "about") {
      setShowAbout(true);
    } else if (key === "fires") {
      onLoadFires();
    } else if (key === "explain" && onExplain) {
      onExplain();
    } else {
      setActivePanel(key);
    }
    // Auto-close drawer on mobile after any action
    onMobileClose?.();
  };

  /* Active state computation ───────────────────── */
  const isActive = (key) =>
    activePanel === key ||
    (key === "declare" && declaring) ||
    (key === "simulation" && simulating) ||
    (key === "risk" && (hasRisk || pickingRisk));

  return (
    /* Mobile: fixed 280px overlay; Desktop: 68px collapsed / 220px expanded */
    <div
      className={[
        "h-full flex flex-col bg-dark-900 border-r border-dark-600/50",
        "transition-[width] duration-300 ease-in-out overflow-hidden z-[1001]",
        "w-[280px]",
        expanded ? "sm:w-[220px]" : "sm:w-[68px]",
      ].join(" ")}
    >
      {/* ── Brand header ──────────────────────────── */}
      <div
        className={[
          "flex-shrink-0 flex items-center border-b border-dark-600/50",
          "px-4 py-4 gap-3",                             // mobile always expanded
          expanded ? "" : "sm:px-0 sm:justify-center sm:gap-0",
        ].join(" ")}
      >
        <img
          src="/logo.svg"
          alt="FSN"
          className="w-9 h-9 flex-shrink-0 drop-shadow-[0_0_6px_rgba(34,211,238,0.3)]"
          draggable={false}
        />
        <div
          className={[
            "flex-1 leading-none overflow-hidden transition-all duration-300",
            expanded ? "sm:opacity-100 sm:w-auto" : "sm:w-0 sm:opacity-0",
          ].join(" ")}
        >
          <div className="text-sm font-black text-white whitespace-nowrap">
            <span className="text-fire-400">FIRE</span>SPREADNET
          </div>
          <div className="text-[9px] text-gray-500 mt-0.5 font-medium whitespace-nowrap">{t("app_subtitle")}</div>
        </div>
        {/* X close — mobile only */}
        <button
          onClick={onMobileClose}
          className="sm:hidden ml-auto flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-gray-500 hover:bg-dark-700 hover:text-gray-200 transition"
          aria-label="Close menu"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* ── Navigation sections ───────────────────── */}
      <nav className="flex-1 overflow-y-auto overflow-x-hidden py-2 px-2 space-y-1 scrollbar-hide">
        {SECTIONS.map(({ group, items }) => (
          <div key={group}>
            {/* Section label: always visible on mobile, hidden on collapsed desktop */}
            <div
              className={[
                "overflow-hidden transition-all duration-200",
                "h-6 opacity-100 mb-0.5",                // mobile always visible
                expanded ? "" : "sm:h-0 sm:opacity-0 sm:mb-0",
              ].join(" ")}
            >
              <p className="text-[9px] font-bold text-gray-600 uppercase tracking-[0.13em] px-2 py-1">
                {t(group)}
              </p>
            </div>

            {/* Items */}
            {items.map(({ key, icon: Icon, label, color }) => {
              const active = isActive(key);
              return (
                <button
                  key={key}
                  onClick={() => handleClick(key)}
                  title={t(label)}
                  className={[
                    "relative group w-full flex items-center gap-3 rounded-xl border",
                    "transition-all duration-150 active:scale-[0.97] min-h-[44px] mb-0.5",
                    "px-3",                              // mobile always padded
                    expanded ? "" : "sm:justify-center sm:px-0",
                    active
                      ? `${color.active} border ring-1 ${color.ring} shadow-sm`
                      : "border-transparent text-gray-500 hover:bg-dark-700/60 hover:text-gray-200 hover:border-dark-500/40",
                  ].join(" ")}
                >
                  <Icon className={`w-[18px] h-[18px] flex-shrink-0 ${active ? "" : "group-hover:scale-110 transition-transform"}`} />

                  {/* Label: always visible on mobile, toggled on desktop */}
                  <span
                    className={[
                      "text-sm font-medium truncate leading-none transition-all duration-300",
                      expanded ? "sm:opacity-100 sm:w-auto" : "sm:opacity-0 sm:w-0 sm:pointer-events-none",
                    ].join(" ")}
                  >
                    {t(label)}
                  </span>

                  {/* ── Badges ───────── */}
                  {key === "fires" && fireCount > 0 && (
                    <span className="ml-auto flex-shrink-0 px-1.5 py-0.5 text-[10px] font-bold bg-red-600 text-white rounded-full leading-none">
                      {fireCount > 99 ? "99+" : fireCount}
                    </span>
                  )}
                  {key === "simulation" && hasSimulation && !simulating && (
                    <span className={`ml-auto w-2 h-2 rounded-full flex-shrink-0 ${color.dot} animate-pulse`} />
                  )}
                  {key === "simulation" && simulating && (
                    <span className={`ml-auto w-2 h-2 rounded-full flex-shrink-0 ${color.dot} animate-ping`} />
                  )}
                  {key === "risk" && (riskLoading || pickingRisk) && (
                    <span className="ml-auto flex-shrink-0 w-3.5 h-3.5 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
                  )}
                  {key === "risk" && hasRisk && !riskLoading && !pickingRisk && (
                    <span className={`ml-auto w-2 h-2 rounded-full flex-shrink-0 ${color.dot} animate-pulse`} />
                  )}
                  {key === "declare" && declaring && (
                    <span className={`ml-auto w-2 h-2 rounded-full flex-shrink-0 ${color.dot} animate-ping`} />
                  )}

                  {/* Tooltip — collapsed desktop only */}
                  {!expanded && (
                    <span className="pointer-events-none absolute left-full ml-3 z-50 hidden sm:block px-2.5 py-1.5 rounded-lg bg-dark-700 border border-dark-500 shadow-xl text-xs text-white font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 translate-x-1 group-hover:translate-x-0 transition-all duration-150">
                      {t(label)}
                    </span>
                  )}
                </button>
              );
            })}

            <div className="h-px bg-dark-700/50 mx-1 mt-1.5 mb-0.5" />
          </div>
        ))}
      </nav>

      {/* ── Status bar ────────────────────────────── */}
      <div
        className={[
          "flex-shrink-0 border-t border-dark-600/50 py-2 px-2",
          expanded ? "" : "sm:flex sm:justify-center",
        ].join(" ")}
      >
        <div
          className={[
            "flex items-center gap-2.5 px-2 py-1.5 rounded-xl",
            expanded ? "" : "sm:justify-center sm:px-0",
          ].join(" ")}
        >
          <div className="relative flex-shrink-0">
            <Activity className="w-4 h-4 text-green-400" />
            <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-green-500 border border-dark-900" />
          </div>
          <div
            className={[
              "overflow-hidden transition-all duration-300",
              expanded ? "sm:opacity-100 sm:w-auto" : "sm:opacity-0 sm:w-0",
            ].join(" ")}
          >
            <div className="text-[10px] font-semibold text-green-400 whitespace-nowrap leading-none">
              {t("sidebar.model_online")}
            </div>
            <div className="text-[9px] text-gray-600 mt-0.5 whitespace-nowrap">PI-CCA · U-Net</div>
          </div>
        </div>
      </div>

      {/* ── Collapse toggle (desktop only) ────────── */}
      <div className="hidden sm:flex flex-shrink-0 border-t border-dark-600/50 p-2">
        <button
          onClick={toggleExpanded}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-xl text-gray-600 hover:bg-dark-700/60 hover:text-gray-300 transition text-xs font-medium"
          title={expanded ? t("sidebar.collapse") : t("sidebar.expand")}
        >
          {expanded ? (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span>{t("sidebar.collapse")}</span>
            </>
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  );
}
