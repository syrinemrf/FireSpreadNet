import { useTranslation } from "react-i18next";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { X, Brain, Info } from "lucide-react";

const FEATURE_COLORS = {
  prev_fire_mask: "#ef4444",
  wind_speed:     "#f97316",
  wind_direction: "#fbbf24",
  max_temp:       "#fb923c",
  min_temp:       "#fdba74",
  humidity:       "#38bdf8",
  drought_index:  "#a78bfa",
  ndvi:           "#4ade80",
  elevation:      "#94a3b8",
  erc:            "#e879f9",
  precipitation:  "#22d3ee",
  population:     "#64748b",
};

const FEATURE_LABELS = {
  prev_fire_mask: "Previous Fire",
  wind_speed:     "Wind Speed",
  wind_direction: "Wind Direction",
  max_temp:       "Max Temp",
  min_temp:       "Min Temp",
  humidity:       "Humidity",
  drought_index:  "Drought Index",
  ndvi:           "Vegetation (NDVI)",
  elevation:      "Elevation",
  erc:            "Energy Release",
  precipitation:  "Precipitation",
  population:     "Population",
};

export default function ExplainPanel({ data, onClose }) {
  const { t } = useTranslation();

  if (!data || !data.factors) return null;

  const isGeneral = data.is_general;

  const chartData = data.factors
    .sort((a, b) => b.importance - a.importance)
    .map((f) => ({
      name: FEATURE_LABELS[f.feature] || f.feature.replace(/_/g, " "),
      value: parseFloat((f.importance * 100).toFixed(1)),
      color: FEATURE_COLORS[f.feature] || "#94a3b8",
    }));

  return (
    <div className="absolute top-14 right-0 bottom-0 w-full sm:w-80 sm:top-16 sm:right-4 sm:bottom-4 z-[1000]">
      <div className="h-full bg-dark-700/95 backdrop-blur-md sm:rounded-2xl border-l sm:border border-dark-400/50 shadow-2xl p-3 sm:p-4 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between mb-3 sm:mb-4">
          <h3 className="text-xs sm:text-sm font-semibold text-white flex items-center gap-2">
            <Brain className="w-4 h-4 text-purple-400" />
            {t("explain.title")}
          </h3>
          <button onClick={onClose} className="text-gray-500 hover:text-white transition">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* General mode badge */}
        {isGeneral && (
          <div className="mb-3 p-2 bg-purple-500/10 border border-purple-500/20 rounded-lg flex items-start gap-2">
            <Info className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
            <p className="text-[10px] sm:text-xs text-purple-300 leading-relaxed">
              {t("explain.general_info")}
            </p>
          </div>
        )}

        <p className="text-[10px] sm:text-xs text-gray-400 mb-3 sm:mb-4">{t("explain.description")}</p>

        {/* Chart */}
        <div className="flex-1 min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 10, top: 4, bottom: 4 }}>
              <XAxis type="number" domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={90} tick={{ fill: "#e2e8f0", fontSize: 9 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1c2538", border: "1px solid #2d3b50", borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: "#e2e8f0" }}
                formatter={(val) => [`${val}%`, t("explain.importance")]}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={16}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Summary */}
        {data.summary && (
          <div className="mt-3 sm:mt-4 p-2 sm:p-3 bg-dark-600 rounded-xl border border-dark-400/50">
            <h4 className="text-[10px] sm:text-xs font-semibold text-gray-300 mb-1">{t("explain.summary")}</h4>
            <p className="text-[10px] sm:text-xs text-gray-400 leading-relaxed">{data.summary}</p>
          </div>
        )}

        {/* Model info */}
        {data.model_name && (
          <div className="mt-2 text-[9px] text-gray-500 text-center">
            Model: <span className="text-gray-400 font-medium">{data.model_name}</span>
          </div>
        )}
      </div>
    </div>
  );
}
