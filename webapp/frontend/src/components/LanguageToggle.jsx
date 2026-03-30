import { useTranslation } from "react-i18next";

export default function LanguageToggle() {
  const { i18n } = useTranslation();
  const lang = i18n.language?.startsWith("fr") ? "fr" : "en";

  const toggle = () => {
    i18n.changeLanguage(lang === "fr" ? "en" : "fr");
  };

  return (
    <button
      onClick={toggle}
      className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-dark-600 border border-dark-400 text-xs font-medium text-gray-300 hover:bg-dark-500 hover:text-white transition"
      title={lang === "fr" ? "Switch to English" : "Passer en Français"}
    >
      <span className={lang === "fr" ? "text-white" : "text-gray-500"}>FR</span>
      <span className="text-gray-600">/</span>
      <span className={lang === "en" ? "text-white" : "text-gray-500"}>EN</span>
    </button>
  );
}
