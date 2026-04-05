import { useTranslation } from "react-i18next";
import { X, ShieldAlert, Thermometer, Droplets, Wind, CloudRain, MapPin, Brain } from "lucide-react";

const RISK_STYLES = {
  extreme: {
    badge: "bg-red-500/20 text-red-300 border-red-500/40",
    bar:   "bg-gradient-to-r from-red-600 to-red-400",
    glow:  "shadow-red-900/30",
    header:"from-red-900/60 to-dark-800",
  },
  high: {
    badge: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    bar:   "bg-gradient-to-r from-amber-600 to-amber-400",
    glow:  "shadow-amber-900/30",
    header:"from-amber-900/40 to-dark-800",
  },
  moderate: {
    badge: "bg-yellow-500/20 text-yellow-300 border-yellow-500/40",
    bar:   "bg-gradient-to-r from-yellow-600 to-yellow-400",
    glow:  "shadow-yellow-900/20",
    header:"from-yellow-900/30 to-dark-800",
  },
  low: {
    badge: "bg-green-500/20 text-green-300 border-green-500/40",
    bar:   "bg-gradient-to-r from-green-600 to-green-400",
    glow:  "shadow-green-900/20",
    header:"from-green-900/20 to-dark-800",
  },
};

const FACTOR_ICONS = {
  temperature: Thermometer,
  humidity: Droplets,
  wind: Wind,
  precipitation: CloudRain,
};

const IMPACT_DOT = {
  high:     "bg-red-500",
  moderate: "bg-amber-500",
  low:      "bg-green-500",
};

const IMPACT_TEXT = {
  high:     "text-red-400",
  moderate: "text-amber-400",
  low:      "text-green-400",
};

export default function RiskPanel({ data, onClose, onSimulate }) {
  const { t } = useTranslation();

  if (!data) return null;

  const level  = data.risk_level || "low";
  const styles = RISK_STYLES[level] || RISK_STYLES.low;
  const riskPct = Math.round((data.risk_score || 0) * 100);

  return (
    /* Full-screen on mobile, side panel on sm+ */
    <div className="fixed inset-0 z-[1100] sm:inset-auto sm:absolute sm:top-0 sm:right-0 sm:bottom-0 sm:w-[340px] flex items-end sm:items-stretch pointer-events-none">
      {/* Mobile backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm sm:hidden pointer-events-auto"
        onClick={onClose}
      />

      {/* Panel */}
      <div className={`relative w-full sm:h-full bg-dark-800/98 backdrop-blur-xl sm:border-l border-t sm:border-t-0 border-dark-500/50 shadow-2xl ${styles.glow} flex flex-col pointer-events-auto max-h-[88vh] sm:max-h-full overflow-hidden rounded-t-3xl sm:rounded-none`}>

        {/* Header gradient */}
        <div className={`flex-shrink-0 bg-gradient-to-b ${styles.header} px-4 pt-5 pb-4 border-b border-dark-600/50`}>
          {/* Source badge — AI not satellite */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-1.5 bg-dark-700/70 rounded-full px-2.5 py-1 border border-amber-500/20">
              <Brain className="w-3 h-3 text-amber-400" />
              <span className="text-[9px] font-bold tracking-widest text-amber-300 uppercase">{t("risk.ai_model_badge")}</span>
            </div>
            <button
              onClick={onClose}
              className="w-7 h-7 flex items-center justify-center rounded-xl hover:bg-white/10 text-gray-400 hover:text-white transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Title */}
          <div className="flex items-start gap-2.5">
            <div className={`mt-0.5 w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 border ${styles.badge}`}>
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-white font-bold text-sm leading-tight">{t("risk.title")}</h3>
              <p className="text-gray-400 text-[10px] mt-0.5 leading-snug">{t("risk.description")}</p>
            </div>
          </div>

          {/* Risk score */}
          <div className={`mt-3 p-3 rounded-xl border ${styles.badge}`}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-bold">{t(`risk.level_${level}`)}</span>
              <span className="text-xl font-black leading-none">{riskPct}%</span>
            </div>
            <div className="w-full h-1.5 bg-dark-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${styles.bar}`}
                style={{ width: `${riskPct}%`, transition: "width 0.6s ease" }}
              />
            </div>
          </div>

          {/* Location */}
          {data.center && (
            <div className="mt-2 flex items-center gap-1.5">
              <MapPin className="w-3 h-3 text-gray-500 flex-shrink-0" />
              <span className="text-[9px] text-gray-500 font-mono">
                {data.center.lat.toFixed(5)}°, {data.center.lon.toFixed(5)}°
              </span>
            </div>
          )}
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto overscroll-contain px-4 py-3 space-y-4">

          {/* Weather conditions */}
          {data.weather && (
            <section>
              <h4 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-2">
                {t("risk.weather_conditions")}
              </h4>
              <div className="grid grid-cols-2 gap-1.5">
                {[
                  { icon: Thermometer, colorCls: "text-fire-400",  key: "factor_temperature", val: data.weather.temperature },
                  { icon: Droplets,    colorCls: "text-blue-400",   key: "factor_humidity",    val: data.weather.humidity },
                  { icon: Wind,        colorCls: "text-gray-300",   key: "factor_wind",        val: data.weather.wind_speed },
                  { icon: CloudRain,   colorCls: "text-cyan-400",   key: "factor_precipitation", val: data.weather.precipitation },
                ].map(({ icon: Icon, colorCls, key, val }) => (
                  <div key={key} className="flex items-center gap-2 bg-dark-700/50 rounded-xl p-2.5">
                    <Icon className={`w-4 h-4 flex-shrink-0 ${colorCls}`} />
                    <div className="min-w-0">
                      <div className="text-[9px] text-gray-500 leading-none">{t(`risk.${key}`)}</div>
                      <div className="text-xs text-white font-semibold mt-0.5 truncate">{val}</div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Contributing factors */}
          {data.factors && data.factors.length > 0 && (
            <section>
              <h4 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-2">
                {t("risk.factors")}
              </h4>
              <div className="space-y-1.5">
                {data.factors.map((f, i) => {
                  const Icon = FACTOR_ICONS[f.factor] || ShieldAlert;
                  return (
                    <div key={i} className="flex items-center gap-2.5 bg-dark-700/40 rounded-xl px-3 py-2">
                      <Icon className="w-3.5 h-3.5 text-gray-500 flex-shrink-0" />
                      <span className="flex-1 text-xs text-gray-300 truncate">
                        {t(`risk.factor_${f.factor}`, { defaultValue: f.factor })}
                      </span>
                      <span className="text-xs text-gray-400 font-mono">{f.value}</span>
                      <div className="flex items-center gap-1">
                        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${IMPACT_DOT[f.impact] || "bg-gray-500"}`} />
                        <span className={`text-[9px] font-medium ${IMPACT_TEXT[f.impact] || "text-gray-400"}`}>
                          {t(`risk.impact_${f.impact}`)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}
        </div>

        {/* CTA */}
        <div className="flex-shrink-0 px-4 pb-5 pt-3 border-t border-dark-600/50">
          {data.center && onSimulate && (
            <button
              onClick={() => onSimulate(data.center.lat, data.center.lon)}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-fire-600 to-fire-700 hover:from-fire-500 hover:to-fire-600 text-white text-xs sm:text-sm font-semibold rounded-xl transition shadow-lg shadow-fire-900/30 active:scale-95"
            >
              <MapPin className="w-4 h-4" />
              {t("simulation.simulate_fire")}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

