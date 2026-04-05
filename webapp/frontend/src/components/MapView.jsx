import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  MapContainer, TileLayer, Marker, Popup, GeoJSON, useMap, useMapEvents,
  CircleMarker, ScaleControl,
} from "react-leaflet";
import L from "leaflet";
import { useTranslation } from "react-i18next";

/* ── FIRMS-style color scale by confidence/brightness ── */
function getFIRMSColor(fire) {
  const conf = String(fire.confidence || "nominal").toLowerCase();
  const brightness = fire.brightness || fire.bright_ti4 || 330;
  if (conf === "high" || brightness > 400) return "#ff0000";
  if (conf === "nominal" || brightness > 350) return "#ff6600";
  if (conf === "low") return "#ffcc00";
  return "#ff6600";
}

function getFIRMSRadius(fire) {
  const frp = fire.frp || 20;
  if (frp > 100) return 8;
  if (frp > 50) return 6;
  if (frp > 20) return 5;
  return 4;
}

/* ── Declared fire icon (blue glow) ── */
const declaredIcon = L.divIcon({
  className: "",
  html: `<div style="width:14px;height:14px;background:radial-gradient(circle,#60a5fa 0%,#3b82f6 60%,transparent 100%);border-radius:50%;box-shadow:0 0 10px 3px rgba(59,130,246,0.5);"></div>`,
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

/* ── Pending fire icon (pulsing yellow ring) ── */
const pendingIcon = L.divIcon({
  className: "",
  html: `<div style="width:18px;height:18px;border:2px solid #fbbf24;border-radius:50%;background:rgba(251,191,36,0.2);animation:fire-pulse 1s ease-in-out infinite;"></div>`,
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

/* ── Re-center map when center/zoom changes ── */
function MapUpdater({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom, { animate: true });
  }, [center, zoom, map]);
  return null;
}

/* ── Handle map clicks for both declaring and simulating ── */
function ClickHandler({ onClick, declaring, simulating, pickingRisk }) {
  useMapEvents({
    click: (e) => {
      if (declaring || simulating || pickingRisk) onClick(e.latlng);
    },
  });
  return null;
}

/* ── Mouse position tracker ── */
function MousePositionTracker({ onMove }) {
  useMapEvents({
    mousemove: (e) => {
      onMove({ lat: e.latlng.lat, lng: e.latlng.lng });
    },
  });
  return null;
}

/* ── Map ref provider ── */
function MapRefProvider({ onRef }) {
  const map = useMap();
  useEffect(() => {
    if (onRef) onRef(map);
  }, [map, onRef]);
  return null;
}

/* ── Simulation overlay styling — FIRMS-like heat gradient ── */
function getSimStyle(hour, feature) {
  const prob = feature?.properties?.probability || 0.5;
  let fillColor;
  if (hour <= 1) fillColor = `rgba(255, 255, 0, ${0.3 + prob * 0.4})`;
  else if (hour <= 3) fillColor = `rgba(255, 200, 0, ${0.3 + prob * 0.4})`;
  else if (hour <= 6) fillColor = `rgba(255, 140, 0, ${0.35 + prob * 0.4})`;
  else if (hour <= 12) fillColor = `rgba(255, 69, 0, ${0.4 + prob * 0.4})`;
  else if (hour <= 18) fillColor = `rgba(220, 38, 38, ${0.45 + prob * 0.4})`;
  else fillColor = `rgba(153, 0, 0, ${0.5 + prob * 0.4})`;

  return {
    color: "transparent",
    weight: 0,
    fillColor: fillColor,
    fillOpacity: 1,
  };
}

/* ── Risk overlay styling — amber/orange gradient ── */
function getRiskStyle(feature) {
  const risk = feature?.properties?.risk || 0;
  const level = feature?.properties?.level || "low";
  let fillColor;
  if (level === "extreme") fillColor = `rgba(220, 38, 38, ${0.25 + risk * 0.35})`;
  else if (level === "high") fillColor = `rgba(245, 158, 11, ${0.2 + risk * 0.35})`;
  else if (level === "moderate") fillColor = `rgba(251, 191, 36, ${0.15 + risk * 0.3})`;
  else fillColor = `rgba(253, 224, 71, ${0.1 + risk * 0.25})`;

  return {
    color: "transparent",
    weight: 0,
    fillColor: fillColor,
    fillOpacity: 1,
  };
}

export default function MapView({
  center,
  zoom,
  fires,
  declaredFires,
  pendingDeclare,
  declaring,
  simulating,
  simGeoJson,
  currentHour,
  onMapClick,
  onLocate,
  onSimulateActive,
  riskGeoJson,
  pickingRisk,
}) {
  const { t } = useTranslation();
  const [mousePos, setMousePos] = useState(null);
  const [currentTime, setCurrentTime] = useState(new Date());
  const mapInstanceRef = useRef(null);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const visibleFrames = useMemo(() => {
    if (!simGeoJson) return [];
    return Object.entries(simGeoJson)
      .filter(([h]) => parseInt(h) <= currentHour)
      .sort(([a], [b]) => parseInt(a) - parseInt(b));
  }, [simGeoJson, currentHour]);

  const handleZoomIn = useCallback(() => {
    mapInstanceRef.current?.zoomIn();
  }, []);

  const handleZoomOut = useCallback(() => {
    mapInstanceRef.current?.zoomOut();
  }, []);

  // Determine cursor style
  const cursorStyle = simulating || declaring || pickingRisk ? "crosshair" : "grab";

  return (
    <div className="flex-1 relative">
      <MapContainer
        center={center}
        zoom={zoom}
        className="w-full h-full z-0"
        zoomControl={false}
        style={{ cursor: cursorStyle }}
      >
        <MapRefProvider onRef={(map) => { mapInstanceRef.current = map; }} />
        <MapUpdater center={center} zoom={zoom} />
        <ClickHandler onClick={onMapClick} declaring={declaring} simulating={simulating} pickingRisk={pickingRisk} />
        <MousePositionTracker onMove={setMousePos} />
        <ScaleControl position="bottomleft" imperial={false} />

        {/* Satellite imagery base — matches NASA FIRMS style */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution='&copy; <a href="https://www.esri.com/">Esri</a> &mdash; Sources: Esri, Maxar, Earthstar Geographics'
          maxZoom={19}
        />
        {/* Semi-transparent labels overlay for readability */}
        <TileLayer
          url="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
          opacity={0.6}
        />

        {/* Active fires — NASA FIRMS style circles */}
        {fires.map((f, i) => (
          <CircleMarker
            key={`fire-${i}`}
            center={[f.latitude, f.longitude]}
            radius={getFIRMSRadius(f)}
            pathOptions={{
              color: getFIRMSColor(f),
              fillColor: getFIRMSColor(f),
              fillOpacity: 0.85,
              weight: 1,
              opacity: 0.9,
            }}
          >
            <Popup>
              <div className="text-sm min-w-[180px]">
                <strong style={{color: getFIRMSColor(f)}}>🔥 {t("fires.active_marker")}</strong>
                <hr style={{borderColor: "#333", margin: "4px 0"}} />
                <div style={{fontSize: "11px", lineHeight: "1.6"}}>
                  <div><span style={{color: "#999"}}>{t("fires.confidence")}:</span> <span style={{color: getFIRMSColor(f), fontWeight: 600}}>{f.confidence}</span></div>
                  <div><span style={{color: "#999"}}>{t("fires.brightness")}:</span> {(f.bright_ti4 || f.brightness || 0).toFixed(1)}K</div>
                  <div><span style={{color: "#999"}}>{t("fires.frp")}:</span> {(f.frp || 0).toFixed(1)} MW</div>
                  <div><span style={{color: "#999"}}>{t("common.position")}:</span> {f.latitude.toFixed(4)}, {f.longitude.toFixed(4)}</div>
                  {f.acq_date && <div><span style={{color: "#999"}}>{t("fires.detected")}:</span> {f.acq_date} {f.acq_time}</div>}
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    // Close popup by clicking the map
                    if (mapInstanceRef.current) mapInstanceRef.current.closePopup();
                    if (onSimulateActive) onSimulateActive(f);
                  }}
                  style={{
                    marginTop: 8, width: "100%", padding: "8px 8px",
                    background: "#ea580c", color: "white", border: "none",
                    borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer",
                    minHeight: 40,
                  }}
                >
                  ▶ {t("simulation.simulate_fire")}
                </button>
              </div>
            </Popup>
          </CircleMarker>
        ))}

        {/* Declared fires */}
        {declaredFires.map((f, i) => (
          <Marker key={`declared-${i}`} position={[f.lat, f.lon]} icon={declaredIcon}>
            <Popup>
              <div className="text-sm">
                <strong className="text-blue-400">{t("fires.declared_marker")}</strong><br />
                {f.lat.toFixed(4)}, {f.lon.toFixed(4)}
              </div>
            </Popup>
          </Marker>
        ))}

        {/* Pending declare marker */}
        {pendingDeclare && (
          <Marker position={[pendingDeclare.lat, pendingDeclare.lon]} icon={pendingIcon}>
            <Popup>{t("declare.click_start_sim")}</Popup>
          </Marker>
        )}

        {/* Risk GeoJSON overlay */}
        {riskGeoJson && riskGeoJson.features && riskGeoJson.features.length > 0 && (
          <GeoJSON
            key={`risk-${riskGeoJson.features.length}`}
            data={riskGeoJson}
            style={getRiskStyle}
            onEachFeature={(feature, layer) => {
              const risk = feature.properties?.risk;
              const level = feature.properties?.level;
              if (risk) {
                layer.bindTooltip(
                  `${t(`risk.level_${level}`)} (${(risk * 100).toFixed(0)}%)`,
                  { permanent: false, direction: "top", className: "sim-tooltip" }
                );
              }
            }}
          />
        )}

        {/* Simulation GeoJSON — realistic fire polygons */}
        {visibleFrames.map(([hour, geojson]) => (
          <GeoJSON
            key={`sim-${hour}-${geojson.features?.length || 0}`}
            data={geojson}
            style={(feature) => getSimStyle(parseInt(hour), feature)}
            onEachFeature={(feature, layer) => {
              const prob = feature.properties?.probability;
              if (prob) {
                layer.bindTooltip(`${(prob * 100).toFixed(0)}%`, {
                  permanent: false, direction: "top", className: "sim-tooltip",
                });
              }
            }}
          />
        ))}
      </MapContainer>

      {/* Simulation badge */}
      {simGeoJson && visibleFrames.length > 0 && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 z-[1000] pointer-events-none">
          <div className="bg-fire-600/90 text-white text-[10px] sm:text-xs font-bold px-3 py-1 rounded-full shadow-lg flex items-center gap-1.5 border border-fire-400/50">
            <span className="inline-block w-2 h-2 rounded-full bg-yellow-300 animate-pulse"></span>
            {t("simulation.badge", { hour: currentHour })}
          </div>
        </div>
      )}

      {/* Map controls (zoom, locate) */}
      <div className="absolute top-20 right-3 z-[1000] flex flex-col gap-1.5">
        <button onClick={handleZoomIn} className="map-ctrl-btn" title="Zoom in">+</button>
        <button onClick={handleZoomOut} className="map-ctrl-btn" title="Zoom out">−</button>
        <button onClick={onLocate} className="map-ctrl-btn" title={t("map.zoom_to_location")}>
          <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>
          </svg>
        </button>
      </div>

      {/* Bottom info bar: Time + Coordinates */}
      <div className="absolute bottom-1 right-1 sm:bottom-2 sm:right-2 z-[1000] flex items-center gap-2 sm:gap-3 bg-dark-800/90 backdrop-blur-sm border border-dark-500/50 rounded-lg px-2 sm:px-3 py-1 sm:py-1.5 text-[9px] sm:text-[10px] text-gray-400 shadow-lg">
        <div className="flex items-center gap-1">
          <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          <span className="font-mono text-gray-300 whitespace-nowrap">
            {currentTime.toLocaleDateString()} {currentTime.toLocaleTimeString()}
          </span>
        </div>
        <div className="w-px h-3 bg-dark-400"></div>
        <div className="font-mono text-gray-300 whitespace-nowrap">
          {mousePos
            ? `${Math.abs(mousePos.lat).toFixed(5)}°${mousePos.lat >= 0 ? "N" : "S"} ${Math.abs(mousePos.lng).toFixed(5)}°${mousePos.lng >= 0 ? "E" : "W"}`
            : "—"
          }
        </div>
      </div>

      {/* FIRMS Legend — satellite-detected fires */}
      {fires.length > 0 && !simGeoJson && (
        <div className="absolute bottom-10 left-1 sm:bottom-14 sm:left-2 z-[1000] bg-dark-800/90 backdrop-blur-sm border border-red-900/50 rounded-lg px-2 sm:px-3 py-2 text-[9px] sm:text-[10px] shadow-lg min-w-[130px]">
          <div className="flex items-center gap-1.5 mb-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></span>
            <span className="text-red-300 font-bold tracking-wide">NASA FIRMS</span>
          </div>
          <div className="text-gray-500 text-[8px] mb-1.5 leading-tight">{t("fires.firms_legend")}</div>
          <div className="space-y-1">
            <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full inline-block flex-shrink-0" style={{background: "#ff0000", boxShadow: "0 0 4px #ff0000"}}></span><span className="text-gray-400">{t("fires.confidence_high")}</span></div>
            <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full inline-block flex-shrink-0" style={{background: "#ff6600", boxShadow: "0 0 4px #ff6600"}}></span><span className="text-gray-400">{t("fires.confidence_nominal")}</span></div>
            <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full inline-block flex-shrink-0" style={{background: "#ffcc00"}}></span><span className="text-gray-400">{t("fires.confidence_low")}</span></div>
          </div>
        </div>
      )}

      {/* Simulation Legend */}
      {simGeoJson && (
        <div className="absolute bottom-10 left-1 sm:bottom-14 sm:left-2 z-[1000] bg-dark-800/90 backdrop-blur-sm border border-dark-500/50 rounded-lg px-2 sm:px-3 py-2 text-[9px] sm:text-[10px] shadow-lg">
          <div className="text-gray-300 font-semibold mb-1.5">🔥 {t("simulation.legend_title")}</div>
          <div className="flex items-center gap-0.5">
            <span className="w-4 h-3 inline-block rounded-sm" style={{background: "rgba(255,255,0,0.7)"}}></span>
            <span className="w-4 h-3 inline-block rounded-sm" style={{background: "rgba(255,200,0,0.7)"}}></span>
            <span className="w-4 h-3 inline-block rounded-sm" style={{background: "rgba(255,140,0,0.7)"}}></span>
            <span className="w-4 h-3 inline-block rounded-sm" style={{background: "rgba(255,69,0,0.7)"}}></span>
            <span className="w-4 h-3 inline-block rounded-sm" style={{background: "rgba(220,38,38,0.7)"}}></span>
            <span className="w-4 h-3 inline-block rounded-sm" style={{background: "rgba(153,0,0,0.8)"}}></span>
          </div>
          <div className="flex justify-between text-gray-500 mt-0.5"><span>0h</span><span>24h</span></div>
        </div>
      )}

      {/* Risk Legend — AI-predicted, clearly distinct from FIRMS */}
      {riskGeoJson && riskGeoJson.features && riskGeoJson.features.length > 0 && !simGeoJson && (
        <div className="absolute bottom-10 left-1 sm:bottom-14 sm:left-2 z-[1000] bg-dark-800/90 backdrop-blur-sm border border-amber-900/50 rounded-lg px-2 sm:px-3 py-2 text-[9px] sm:text-[10px] shadow-lg min-w-[130px]">
          <div className="flex items-center gap-1.5 mb-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span>
            <span className="text-amber-300 font-bold tracking-wide">{t("risk.ai_model_badge")}</span>
          </div>
          <div className="text-gray-500 text-[8px] mb-1.5 leading-tight">{t("risk.legend")}</div>
          <div className="space-y-1">
            <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-sm inline-block flex-shrink-0" style={{background: "rgba(220,38,38,0.7)"}}></span><span className="text-gray-400">{t("risk.level_extreme")}</span></div>
            <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-sm inline-block flex-shrink-0" style={{background: "rgba(245,158,11,0.6)"}}></span><span className="text-gray-400">{t("risk.level_high")}</span></div>
            <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-sm inline-block flex-shrink-0" style={{background: "rgba(251,191,36,0.5)"}}></span><span className="text-gray-400">{t("risk.level_moderate")}</span></div>
            <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-sm inline-block flex-shrink-0" style={{background: "rgba(253,224,71,0.35)"}}></span><span className="text-gray-400">{t("risk.level_low")}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
