import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Flame, ShieldAlert, Play, X } from "lucide-react";

export default function WelcomeGuide({ onDismiss }) {
  const { t } = useTranslation();
  const [dontShow, setDontShow] = useState(false);

  return (
    <div className="fixed inset-0 z-[3000] flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={() => onDismiss(dontShow)} />

      {/* Modal */}
      <div className="relative w-full max-w-md bg-dark-700 rounded-2xl border border-dark-400/50 shadow-2xl overflow-hidden">
        {/* Header with gradient */}
        <div className="relative bg-gradient-to-br from-fire-600/30 to-dark-700 px-6 py-5 border-b border-dark-500/50">
          <button
            onClick={() => onDismiss(dontShow)}
            className="absolute top-3 right-3 text-gray-500 hover:text-white transition p-1"
          >
            <X className="w-4 h-4" />
          </button>
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-fire-500 to-fire-700 flex items-center justify-center text-lg">
              🔥
            </div>
            <div>
              <h2 className="text-base font-bold text-white">{t("welcome.title")}</h2>
              <p className="text-[11px] text-gray-400">{t("welcome.subtitle")}</p>
            </div>
          </div>
        </div>

        {/* Features */}
        <div className="px-6 py-5 space-y-4">
          {/* Active Fires */}
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-red-600/20 flex items-center justify-center flex-shrink-0">
              <Flame className="w-4 h-4 text-red-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">{t("welcome.layer_fires")}</h3>
              <p className="text-xs text-gray-400 leading-relaxed">{t("welcome.layer_fires_desc")}</p>
            </div>
          </div>

          {/* Fire Risk */}
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-amber-600/20 flex items-center justify-center flex-shrink-0">
              <ShieldAlert className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">{t("welcome.layer_risk")}</h3>
              <p className="text-xs text-gray-400 leading-relaxed">{t("welcome.layer_risk_desc")}</p>
            </div>
          </div>

          {/* Simulation */}
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-green-600/20 flex items-center justify-center flex-shrink-0">
              <Play className="w-4 h-4 text-green-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">{t("welcome.layer_sim")}</h3>
              <p className="text-xs text-gray-400 leading-relaxed">{t("welcome.layer_sim_desc")}</p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-dark-800/50 border-t border-dark-500/50 flex items-center justify-between">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={dontShow}
              onChange={(e) => setDontShow(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-dark-400 bg-dark-600 text-fire-500 focus:ring-fire-500/30"
            />
            <span className="text-xs text-gray-400">{t("welcome.dont_show")}</span>
          </label>
          <button
            onClick={() => onDismiss(dontShow)}
            className="px-5 py-2 bg-fire-600 hover:bg-fire-500 text-white text-sm font-medium rounded-lg transition"
          >
            {t("welcome.got_it")}
          </button>
        </div>
      </div>
    </div>
  );
}
