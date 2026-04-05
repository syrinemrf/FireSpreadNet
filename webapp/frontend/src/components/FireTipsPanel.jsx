import { useTranslation } from "react-i18next";
import { X, AlertTriangle, Phone, Wind, MapPin, Zap, ShieldCheck } from "lucide-react";

/* Priority-ordered emergency steps */
const TIPS = [
  {
    key: "call_emergency",
    icon: Phone,
    num: "1",
    accent: { ring: "ring-red-500/60",     icon: "bg-red-500/25 text-red-200",     title: "text-red-200",   card: "bg-red-950/60 border-red-700/40" },
  },
  {
    key: "evacuate",
    icon: Wind,
    num: "2",
    accent: { ring: "ring-amber-500/60",   icon: "bg-amber-500/25 text-amber-200", title: "text-amber-200", card: "bg-amber-950/60 border-amber-700/40" },
  },
  {
    key: "alert_neighbors",
    icon: AlertTriangle,
    num: "3",
    accent: { ring: "ring-orange-500/60",  icon: "bg-orange-500/25 text-orange-200", title: "text-orange-200", card: "bg-orange-950/60 border-orange-700/40" },
  },
  {
    key: "no_vehicles",
    icon: Zap,
    num: "4",
    accent: { ring: "ring-yellow-500/60",  icon: "bg-yellow-500/25 text-yellow-200", title: "text-yellow-200", card: "bg-yellow-950/50 border-yellow-700/40" },
  },
  {
    key: "stay_informed",
    icon: ShieldCheck,
    num: "5",
    accent: { ring: "ring-green-500/60",   icon: "bg-green-500/25 text-green-200",  title: "text-green-200",  card: "bg-green-950/50 border-green-700/40" },
  },
];

export default function FireTipsPanel({ location, onSimulate, onClose }) {
  const { t } = useTranslation();

  return (
    <div className="fixed inset-0 z-[1500] sm:inset-auto sm:top-0 sm:right-0 sm:bottom-0 sm:w-[340px] flex items-end sm:items-stretch pointer-events-none">
      {/* Mobile backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm sm:hidden pointer-events-auto"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="relative w-full sm:h-full bg-[#0f1117] sm:border-l border-t sm:border-t-0 border-red-900/40 shadow-2xl shadow-red-950/50 flex flex-col pointer-events-auto max-h-[92vh] sm:max-h-full overflow-hidden rounded-t-3xl sm:rounded-none">

        {/* ── Header ───────────────────────────────────────── */}
        <div className="flex-shrink-0 bg-gradient-to-br from-red-950/90 via-red-900/50 to-[#0f1117] px-4 pt-4 pb-4 border-b border-red-800/30">
          {/* Top row: pulse badge + close */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500" />
              </span>
              <span className="text-red-300 text-[11px] font-black tracking-[0.2em] uppercase">
                {t("tips.emergency")}
              </span>
            </div>
            <button
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center rounded-xl bg-white/5 hover:bg-white/15 text-gray-300 hover:text-white transition"
              aria-label="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Title & subtitle */}
          <h2 className="text-white font-black text-base leading-tight">
            {t("tips.title")}
          </h2>
          <p className="text-red-200 text-xs mt-1 leading-relaxed opacity-90">
            {t("tips.subtitle")}
          </p>

          {/* GPS coords */}
          {location && (
            <div className="mt-2.5 flex items-center gap-2 bg-black/40 rounded-lg px-3 py-2 border border-white/5">
              <MapPin className="w-3.5 h-3.5 text-fire-400 flex-shrink-0" />
              <span className="text-[11px] text-gray-200 font-mono tracking-wide">
                {location.lat.toFixed(5)}°, {location.lon.toFixed(5)}°
              </span>
            </div>
          )}
        </div>

        {/* ── Scrollable tips ──────────────────────────────── */}
        <div className="flex-1 overflow-y-auto overscroll-contain px-3 py-3 space-y-2">
          {TIPS.map(({ key, icon: Icon, num, accent }) => (
            <div
              key={key}
              className={`flex items-start gap-3 p-3 rounded-2xl border ${accent.card}`}
            >
              {/* Number + icon badge */}
              <div className="flex-shrink-0 flex flex-col items-center gap-1">
                <div className={`w-9 h-9 rounded-xl flex items-center justify-center ring-1 ${accent.ring} ${accent.icon}`}>
                  <Icon className="w-4.5 h-4.5" />
                </div>
                <span className="text-[9px] font-bold text-gray-600">#{num}</span>
              </div>

              {/* Text */}
              <div className="min-w-0 pt-0.5">
                <p className={`text-[13px] font-bold leading-snug ${accent.title} mb-1`}>
                  {t(`tips.${key}_title`)}
                </p>
                <p className="text-[11px] text-gray-200 leading-relaxed opacity-90">
                  {t(`tips.${key}_desc`)}
                </p>
              </div>
            </div>
          ))}

          {/* Emergency numbers */}
          <div className="p-3 rounded-2xl bg-dark-800/80 border border-dark-600/50">
            <p className="text-[10px] font-bold text-gray-300 uppercase tracking-widest mb-2.5">
              {t("tips.emergency_numbers")}
            </p>
            <div className="grid grid-cols-3 gap-2">
              {[
                { label: "112", desc: "EU",   color: "from-red-700 to-red-800" },
                { label: "18",  desc: "FR",   color: "from-orange-700 to-orange-800" },
                { label: "15",  desc: "SAMU", color: "from-amber-700 to-amber-800" },
              ].map((n) => (
                <a
                  key={n.label}
                  href={`tel:${n.label}`}
                  className={`flex flex-col items-center justify-center py-2.5 bg-gradient-to-b ${n.color} rounded-xl border border-white/10 hover:brightness-110 active:scale-95 transition`}
                >
                  <span className="text-white font-black text-lg leading-none">{n.label}</span>
                  <span className="text-[9px] text-white/60 mt-0.5 font-medium">{n.desc}</span>
                </a>
              ))}
            </div>
          </div>
        </div>

        {/* ── CTA footer ───────────────────────────────────── */}
        <div className="flex-shrink-0 px-3 pb-4 pt-3 border-t border-dark-700/70 space-y-2 bg-[#0f1117]">
          <button
            onClick={onSimulate}
            className="w-full flex items-center justify-center gap-2.5 px-4 py-3.5 bg-gradient-to-r from-fire-500 to-fire-600 hover:from-fire-400 hover:to-fire-500 text-white text-sm font-bold rounded-2xl transition shadow-lg shadow-fire-900/50 active:scale-[0.98]"
          >
            <Zap className="w-4.5 h-4.5" />
            {t("tips.simulate_cta")}
          </button>
          <button
            onClick={onClose}
            className="w-full py-2 text-gray-500 hover:text-gray-300 text-xs font-medium transition"
          >
            {t("tips.close_panel")}
          </button>
        </div>
      </div>
    </div>
  );
}
