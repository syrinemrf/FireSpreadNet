import { useTranslation } from "react-i18next";
import { X, Flame, Database, AlertTriangle, Code } from "lucide-react";

export default function AboutModal({ onClose }) {
  const { t } = useTranslation();

  const Section = ({ icon: Icon, title, children, iconColor = "text-fire-400" }) => (
    <div className="mb-5">
      <h3 className={`text-sm font-semibold flex items-center gap-2 mb-2 ${iconColor}`}>
        <Icon className="w-4 h-4" /> {title}
      </h3>
      <div className="text-xs text-gray-400 leading-relaxed">{children}</div>
    </div>
  );

  return (
    <div className="fixed inset-0 z-[2000] flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative w-full max-w-lg max-h-[90vh] sm:max-h-[80vh] bg-dark-700 rounded-xl sm:rounded-2xl border border-dark-400/50 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="sticky top-0 bg-dark-700 border-b border-dark-500/50 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-fire-500 to-fire-700 flex items-center justify-center text-lg">
              🔥
            </div>
            <div>
              <h2 className="text-base font-bold text-white">{t("about.title")}</h2>
              <p className="text-[10px] text-gray-400">{t("app_subtitle")}</p>
            </div>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white transition p-1">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-5 overflow-y-auto max-h-[calc(80vh-72px)]">
          <Section icon={Flame} title={t("about.what")}>
            <p>{t("about.what_text")}</p>
          </Section>

          <Section icon={Code} title={t("about.how")} iconColor="text-blue-400">
            <p>{t("about.how_text")}</p>
          </Section>

          <Section icon={Database} title={t("about.data_sources")} iconColor="text-green-400">
            <ul className="space-y-1.5 list-none">
              <li className="flex items-start gap-2"><span className="text-fire-400 mt-0.5">•</span>{t("about.data_firms")}</li>
              <li className="flex items-start gap-2"><span className="text-blue-400 mt-0.5">•</span>{t("about.data_weather")}</li>
              <li className="flex items-start gap-2"><span className="text-gray-400 mt-0.5">•</span>{t("about.data_elevation")}</li>
            </ul>
          </Section>

          <Section icon={AlertTriangle} title={t("about.limitations")} iconColor="text-yellow-400">
            <p className="p-2 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">{t("about.limitations_text")}</p>
          </Section>

          <Section icon={Code} title={t("about.tech_stack")} iconColor="text-purple-400">
            <p>{t("about.tech_text")}</p>
          </Section>

          <div className="border-t border-dark-500/50 pt-4 mt-4">
            <p className="text-[10px] text-gray-500">{t("about.credits_text")}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
