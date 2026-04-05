import { useState, useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { Play, Pause, SkipForward, RotateCcw, Brain, Loader } from "lucide-react";

export default function SimulationPanel({
  simulation,
  currentHour,
  setCurrentHour,
  onStep,
  onExplain,
  onReset,
  loading,
}) {
  const { t } = useTranslation();
  const [autoPlay, setAutoPlay] = useState(false);
  const intervalRef = useRef(null);

  const maxHour = simulation ? Math.max(...Object.keys(simulation.frames || {}).map(Number), 0) : 0;

  useEffect(() => {
    if (autoPlay && currentHour < 24) {
      intervalRef.current = setInterval(() => {
        onStep(1);
      }, 1500);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoPlay, currentHour, onStep]);

  useEffect(() => {
    if (currentHour >= 24 && autoPlay) setAutoPlay(false);
  }, [currentHour, autoPlay]);

  const totalFrames = Object.keys(simulation?.frames || {}).length;
  const area = simulation?.burned_area_km2;

  // Weather info from simulation
  const weather = simulation?.weather;

  return (
    <div className="absolute bottom-2 left-1/2 -translate-x-1/2 z-[1000] w-[95vw] sm:w-[600px] max-w-[calc(100vw-1rem)]">
      <div className="bg-dark-700/95 backdrop-blur-md rounded-2xl border border-dark-400/50 shadow-2xl p-2.5 sm:p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-2 sm:mb-3">
          <h3 className="text-xs sm:text-sm font-semibold text-white flex items-center gap-2">
            <Play className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-fire-400" />
            {t("simulation.title")}
          </h3>
          <div className="flex items-center gap-2 sm:gap-3 text-[9px] sm:text-xs text-gray-400">
            {area != null && (
              <span>
                {t("simulation.total_area")}: <span className="text-fire-400 font-medium">{area.toFixed(1)} {t("simulation.km2")}</span>
              </span>
            )}
          </div>
        </div>

        {/* Time slider */}
        <div className="mb-2 sm:mb-3">
          <input
            type="range"
            min={0}
            max={Math.max(maxHour, 1)}
            value={currentHour}
            onChange={(e) => setCurrentHour(parseInt(e.target.value))}
            className="w-full h-1.5 bg-dark-500 rounded-lg appearance-none cursor-pointer accent-fire-500"
          />
          <div className="flex justify-between text-[9px] sm:text-[10px] text-gray-500 mt-1">
            <span>0h</span>
            <span className="text-fire-400 font-medium">
              H+{currentHour} / {maxHour > 0 ? maxHour : "..."} ({totalFrames} frames)
            </span>
            <span>24h</span>
          </div>
        </div>

        {/* Controls — responsive grid */}
        <div className="flex items-center justify-center gap-1 sm:gap-2 flex-wrap">
          <button
            onClick={() => onStep(1)}
            disabled={loading || currentHour >= 24}
            className="flex items-center gap-1 px-2.5 sm:px-3 py-2 sm:py-1.5 bg-fire-600 hover:bg-fire-500 disabled:bg-dark-500 disabled:text-gray-600 text-white text-[10px] sm:text-xs rounded-lg transition min-h-[36px] sm:min-h-0"
          >
            {loading ? <Loader className="w-3 h-3 sm:w-3.5 sm:h-3.5 animate-spin" /> : <SkipForward className="w-3 h-3 sm:w-3.5 sm:h-3.5" />}
            +1h
          </button>

          <button
            onClick={() => onStep(3)}
            disabled={loading || currentHour >= 24}
            className="flex items-center gap-1 px-2.5 sm:px-3 py-2 sm:py-1.5 bg-fire-700 hover:bg-fire-600 disabled:bg-dark-500 disabled:text-gray-600 text-white text-[10px] sm:text-xs rounded-lg transition min-h-[36px] sm:min-h-0"
          >
            +3h
          </button>

          <button
            onClick={() => onStep(6)}
            disabled={loading || currentHour >= 24}
            className="flex items-center gap-1 px-2.5 sm:px-3 py-2 sm:py-1.5 bg-fire-700 hover:bg-fire-600 disabled:bg-dark-500 disabled:text-gray-600 text-white text-[10px] sm:text-xs rounded-lg transition min-h-[36px] sm:min-h-0"
          >
            +6h
          </button>

          <button
            onClick={() => setAutoPlay(!autoPlay)}
            disabled={currentHour >= 24}
            className={`flex items-center gap-1 px-2.5 sm:px-3 py-2 sm:py-1.5 text-[10px] sm:text-xs rounded-lg transition min-h-[36px] sm:min-h-0 ${
              autoPlay
                ? "bg-yellow-600 hover:bg-yellow-500 text-white"
                : "bg-dark-500 hover:bg-dark-400 text-gray-300"
            }`}
          >
            {autoPlay ? <Pause className="w-3 h-3 sm:w-3.5 sm:h-3.5" /> : <Play className="w-3 h-3 sm:w-3.5 sm:h-3.5" />}
            <span className="hidden sm:inline">{autoPlay ? t("simulation.pause") : t("simulation.auto_play")}</span>
          </button>

          <div className="w-px h-5 bg-dark-400 mx-0.5 hidden sm:block" />

          <button
            onClick={onExplain}
            disabled={loading}
            className="flex items-center gap-1 px-2.5 sm:px-3 py-2 sm:py-1.5 bg-purple-600/20 hover:bg-purple-600/30 text-purple-400 text-[10px] sm:text-xs rounded-lg border border-purple-500/30 transition min-h-[36px] sm:min-h-0"
          >
            <Brain className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
            <span className="hidden sm:inline">{t("simulation.explain")}</span>
          </button>

          <button
            onClick={onReset}
            className="flex items-center gap-1 px-2.5 sm:px-3 py-2 sm:py-1.5 bg-dark-500 hover:bg-dark-400 text-gray-400 text-[10px] sm:text-xs rounded-lg transition min-h-[36px] sm:min-h-0"
          >
            <RotateCcw className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
            <span className="hidden sm:inline">{t("simulation.reset")}</span>
          </button>
        </div>

        {/* Loading overlay */}
        {loading && (
          <div className="mt-2 flex items-center justify-center gap-2 text-[10px] sm:text-xs text-fire-400">
            <Loader className="w-3 h-3 animate-spin" />
            {t("simulation.computing")}
          </div>
        )}
      </div>
    </div>
  );
}
