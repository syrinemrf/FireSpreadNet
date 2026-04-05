import { useState, useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import MapView from "./components/MapView";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import SimulationPanel from "./components/SimulationPanel";
import ExplainPanel from "./components/ExplainPanel";
import RiskPanel from "./components/RiskPanel";
import FireTipsPanel from "./components/FireTipsPanel";
import AboutModal from "./components/AboutModal";
import WelcomeGuide from "./components/WelcomeGuide";
import api from "./services/api";
import useGeolocation from "./hooks/useGeolocation";

export default function App() {
  const { t } = useTranslation();
  const { position: geoPos, error: geoError, locate } = useGeolocation();

  /* ── state ─────────────────────────────── */
  const [fires, setFires] = useState([]);
  const [declaredFires, setDeclaredFires] = useState([]);
  const [declaring, setDeclaring] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [pendingDeclare, setPendingDeclare] = useState(null);
  const [simulation, setSimulation] = useState(null);
  const [simGeoJson, setSimGeoJson] = useState(null);
  const [currentHour, setCurrentHour] = useState(0);
  const [explainData, setExplainData] = useState(null);
  const [riskData, setRiskData] = useState(null);
  const [riskLoading, setRiskLoading] = useState(false);
  const [pickingRisk, setPickingRisk] = useState(false);
  const [showFireTips, setShowFireTips] = useState(false);
  const [fireTipsLocation, setFireTipsLocation] = useState(null);
  const [activePanel, setActivePanel] = useState("fires");
  const [showAbout, setShowAbout] = useState(false);
  const [showWelcome, setShowWelcome] = useState(() => {
    return !localStorage.getItem("fsn_welcome_dismissed");
  });
  const [mapCenter, setMapCenter] = useState([20, -10]);
  const [mapZoom, setMapZoom] = useState(3);
  const [loading, setLoading] = useState(false);
  const [simLoading, setSimLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  /* ── fetch active fires ────────────────── */
  const loadFires = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await api.get("/api/fires/active");
      setFires(resp.data.fires || []);
    } catch {
      setFires([]);
    } finally {
      setLoading(false);
    }
  }, []);

  /* ── auto-load fires on mount ──────────── */
  useEffect(() => {
    loadFires();
  }, [loadFires]);

  /* ── map click handling ─────────────────── */
  const handleMapClick = useCallback(
    (latlng) => {
      if (simulating) {
        startSimulation(latlng.lat, latlng.lng, 1.0);
        return;
      }
      if (declaring) {
        setPendingDeclare({ lat: latlng.lat, lon: latlng.lng, radius_km: 1.0 });
        return;
      }
      if (pickingRisk) {
        setPickingRisk(false);
        setRiskLoading(true);
        setErrorMsg(null);
        // Fly to the picked location at ~1km altitude (zoom 15)
        setMapCenter([latlng.lat, latlng.lng]);
        setMapZoom(15);
        api.post("/api/fires/risk", {
            latitude: latlng.lat,
            longitude: latlng.lng,
            radius_km: 32,
          }).then((resp) => {
            setRiskData(resp.data);
            setActivePanel("risk");
            setSidebarOpen(false);
          }).catch((err) => {
            const msg = err.response?.data?.detail || err.message;
            setErrorMsg(msg);
            setTimeout(() => setErrorMsg(null), 5000);
          }).finally(() => setRiskLoading(false));
        return;
      }
    },
    [declaring, simulating, pickingRisk]
  );

  const confirmDeclare = useCallback(async () => {
    if (!pendingDeclare) return;
    setLoading(true);
    try {
      const resp = await api.post("/api/fires/declare", {
        latitude: pendingDeclare.lat,
        longitude: pendingDeclare.lon,
        lat: pendingDeclare.lat,
        lon: pendingDeclare.lon,
        radius_km: pendingDeclare.radius_km,
      });
      const newFire = { ...resp.data.fire, lat: pendingDeclare.lat, lon: pendingDeclare.lon };
      setDeclaredFires((prev) => [...prev, newFire]);
      // Fly to declared fire location at 1km altitude (zoom 15)
      setMapCenter([pendingDeclare.lat, pendingDeclare.lon]);
      setMapZoom(15);
      // Show emergency tips panel
      setFireTipsLocation({ lat: pendingDeclare.lat, lon: pendingDeclare.lon });
      setShowFireTips(true);
      setPendingDeclare(null);
      setDeclaring(false);
    } catch (err) {
      console.error("Declare error:", err);
    } finally {
      setLoading(false);
    }
  }, [pendingDeclare]);

  /* ── simulation ────────────────────────── */
  const startSimulation = useCallback(async (lat, lon, radius_km = 1.0) => {
    setSimLoading(true);
    setErrorMsg(null);
    try {
      const resp = await api.post("/api/simulation/start", {
        latitude: lat,
        longitude: lon,
        radius_km: radius_km,
      });
      setSimulation(resp.data);
      setSimGeoJson(resp.data.frames);
      setCurrentHour(resp.data.current_hour);
      setActivePanel("simulation");
      setPendingDeclare(null);
      setDeclaring(false);
      setSimulating(false);
      setSidebarOpen(false);
      setRiskData(null);
      // Center map on simulation
      setMapCenter([lat, lon]);
      setMapZoom(11);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message;
      setErrorMsg(msg);
      setTimeout(() => setErrorMsg(null), 5000);
      console.error("Simulation start error:", err);
    } finally {
      setSimLoading(false);
    }
  }, []);

  const startSimFromPending = useCallback(async () => {
    if (!pendingDeclare && declaredFires.length === 0) return;
    const target = pendingDeclare || declaredFires[declaredFires.length - 1];
    await startSimulation(target.lat, target.lon, target.radius_km || 1.0);
  }, [pendingDeclare, declaredFires, startSimulation]);

  const simulateActiveFire = useCallback(async (fire) => {
    await startSimulation(fire.latitude, fire.longitude, 1.0);
  }, [startSimulation]);

  const stepSimulation = useCallback(
    async (hours = 1) => {
      if (!simulation) return;
      setSimLoading(true);
      try {
        const resp = await api.post(`/api/simulation/${simulation.simulation_id}/step`, { hours });
        setSimulation((prev) => ({ ...prev, ...resp.data }));
        setSimGeoJson(resp.data.frames);
        setCurrentHour(resp.data.current_hour);
      } catch (err) {
        console.error("Step error:", err);
        const msg = err.response?.data?.detail || err.message;
        setErrorMsg(t("step_error", { defaultValue: "Erreur lors de l'avancement : " + msg }));
        setTimeout(() => setErrorMsg(null), 5000);
      } finally {
        setSimLoading(false);
      }
    },
    [simulation]
  );

  const fetchExplainability = useCallback(async () => {
    setLoading(true);
    try {
      let resp;
      if (simulation) {
        resp = await api.get(`/api/explainability/${simulation.simulation_id}`);
      } else {
        resp = await api.get("/api/explainability/general");
      }
      setExplainData(resp.data);
      setActivePanel("explain");
    } catch (err) {
      console.error("Explain error:", err);
    } finally {
      setLoading(false);
    }
  }, [simulation]);

  const fetchRisk = useCallback(async () => {
    // Enter picking mode — user clicks on the map to choose location
    setPickingRisk(true);
    if (declaring) setDeclaring(false);
    if (simulating) setSimulating(false);
    setSidebarOpen(false);
  }, [declaring, simulating]);

  const clearRisk = useCallback(() => {
    setRiskData(null);
    setPickingRisk(false);
    setActivePanel("fires");
  }, []);

  const resetSimulation = useCallback(() => {
    setSimulation(null);
    setSimGeoJson(null);
    setCurrentHour(0);
    setExplainData(null);
    setActivePanel("fires");
  }, []);

  /* ── geolocation ───────────────────────── */
  useEffect(() => {
    if (geoPos) {
      setMapCenter([geoPos.lat, geoPos.lon]);
      setMapZoom(15);
    }
  }, [geoPos]);

  useEffect(() => {
    if (geoError) {
      setErrorMsg(t("geo_error", { defaultValue: "Localisation impossible : " + geoError }));
      setTimeout(() => setErrorMsg(null), 5000);
    }
  }, [geoError, t]);

  const handleLocate = useCallback(() => {
    locate();
  }, [locate]);

  /* ── search location ───────────────────── */
  const handleSearch = useCallback(async (query) => {
    try {
      const resp = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`
      );
      const data = await resp.json();
      if (data.length > 0) {
        setMapCenter([parseFloat(data[0].lat), parseFloat(data[0].lon)]);
        setMapZoom(13);
      }
    } catch {
      /* ignore geocoding errors */
    }
  }, []);

  const dismissWelcome = useCallback((dontShowAgain) => {
    setShowWelcome(false);
    if (dontShowAgain) {
      localStorage.setItem("fsn_welcome_dismissed", "1");
    }
  }, []);

  /* ── active interaction mode label ─────── */
  const interactionMode = simulating ? "simulation" : declaring ? "declare" : pickingRisk ? "risk" : null;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-dark-900">
      {/* Sidebar — slides in from left on mobile, always visible on desktop */}
      <div className={`${sidebarOpen ? "translate-x-0" : "-translate-x-full"} sm:translate-x-0 fixed sm:relative z-[1050] h-full transition-transform duration-300 ease-in-out`}>
        <Sidebar
          activePanel={activePanel}
          setActivePanel={(p) => { setActivePanel(p); setSidebarOpen(false); }}
          declaring={declaring}
          setDeclaring={setDeclaring}
          simulating={simulating}
          setSimulating={setSimulating}
          setShowAbout={setShowAbout}
          onLoadFires={loadFires}
          onExplain={fetchExplainability}
          onRisk={fetchRisk}
          onClearRisk={clearRisk}
          fireCount={fires.length}
          hasSimulation={!!simulation}
          hasRisk={!!riskData}
          riskLoading={riskLoading}
          pickingRisk={pickingRisk}
          onMobileClose={() => setSidebarOpen(false)}
        />
      </div>

      {/* Mobile backdrop — tap outside to close */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/60 z-[1040] sm:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Main area */}
      <div className="flex-1 flex flex-col relative">
        <TopBar
          onSearch={handleSearch}
          onMenuToggle={() => setSidebarOpen(!sidebarOpen)}
          sidebarOpen={sidebarOpen}
        />

        {/* Map */}
        <MapView
          center={mapCenter}
          zoom={mapZoom}
          fires={fires}
          declaredFires={declaredFires}
          pendingDeclare={pendingDeclare}
          declaring={declaring}
          simulating={simulating}
          simGeoJson={simGeoJson}
          currentHour={currentHour}
          onMapClick={handleMapClick}
          onLocate={handleLocate}
          onSimulateActive={simulateActiveFire}
          riskGeoJson={riskData?.geojson}
          pickingRisk={pickingRisk}
        />

        {/* Loading indicator */}
        {loading && fires.length === 0 && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 z-[1200] bg-dark-700/95 backdrop-blur text-white text-xs sm:text-sm px-4 py-2 rounded-xl shadow-lg border border-dark-400/50 flex items-center gap-2">
            <span className="inline-block w-3 h-3 border-2 border-fire-400 border-t-transparent rounded-full animate-spin"></span>
            {t("fires.loading")}
          </div>
        )}

        {/* Simulation loading overlay */}
        {simLoading && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 z-[1200] bg-dark-700/95 backdrop-blur text-white text-xs sm:text-sm px-4 py-2 rounded-xl shadow-lg border border-fire-500/50 flex items-center gap-2">
            <span className="inline-block w-3 h-3 border-2 border-fire-400 border-t-transparent rounded-full animate-spin"></span>
            {t("simulation.running")}
          </div>
        )}

        {/* Risk loading overlay */}
        {riskLoading && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 z-[1200] bg-dark-700/95 backdrop-blur text-white text-xs sm:text-sm px-4 py-2 rounded-xl shadow-lg border border-amber-500/50 flex items-center gap-2">
            <span className="inline-block w-3 h-3 border-2 border-amber-400 border-t-transparent rounded-full animate-spin"></span>
            {t("risk.loading")}
          </div>
        )}

        {/* Error toast */}
        {errorMsg && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 z-[1200] bg-red-600/95 text-white text-xs sm:text-sm px-4 py-2 rounded-xl shadow-lg border border-red-400/50 max-w-[90vw] text-center">
            ⚠️ {errorMsg}
          </div>
        )}

        {/* Simulation pick mode overlay */}
        {simulating && (
          <div className="absolute top-16 left-1/2 -translate-x-1/2 z-[1000] bg-dark-700/95 backdrop-blur px-4 sm:px-6 py-2 sm:py-3 rounded-xl border border-green-500/30 shadow-lg flex flex-wrap items-center justify-center gap-2 sm:gap-4 max-w-[95vw]">
            <span className="text-green-400 text-xs sm:text-sm font-medium">{t("simulation.click_to_start")}</span>
            <button onClick={() => setSimulating(false)} className="text-gray-400 hover:text-white text-sm">✕</button>
          </div>
        )}

        {/* Risk pick mode overlay */}
        {pickingRisk && (
          <div className="absolute top-16 left-1/2 -translate-x-1/2 z-[1000] bg-dark-700/95 backdrop-blur px-4 sm:px-6 py-2 sm:py-3 rounded-xl border border-amber-500/30 shadow-lg flex flex-wrap items-center justify-center gap-2 sm:gap-4 max-w-[95vw]">
            <span className="text-amber-300 text-xs sm:text-sm font-medium">⚑ {t("risk.click_to_pick")}</span>
            <button onClick={() => setPickingRisk(false)} className="text-gray-400 hover:text-white text-sm">✕</button>
          </div>
        )}

        {/* Declare overlay */}
        {declaring && (
          <div className="absolute top-16 left-1/2 -translate-x-1/2 z-[1000] bg-dark-700/95 backdrop-blur px-4 sm:px-6 py-2 sm:py-3 rounded-xl border border-fire-500/30 shadow-lg flex flex-wrap items-center justify-center gap-2 sm:gap-4 max-w-[95vw]">
            <span className="text-fire-400 text-xs sm:text-sm font-medium">{t("map.click_to_declare")}</span>
            {pendingDeclare && (
              <>
                <span className="text-[10px] sm:text-xs text-gray-400">
                  {pendingDeclare.lat.toFixed(4)}, {pendingDeclare.lon.toFixed(4)}
                </span>
                <button onClick={confirmDeclare} className="px-3 py-1 bg-fire-600 text-white rounded-lg text-xs hover:bg-fire-500 transition">
                  ▶ {t("declare.confirm")}
                </button>
              </>
            )}
            <button onClick={() => { setDeclaring(false); setPendingDeclare(null); }} className="text-gray-400 hover:text-white text-sm">
              ✕
            </button>
          </div>
        )}

        {/* Simulation panel */}
        {activePanel === "simulation" && simulation && (
          <SimulationPanel
            simulation={simulation}
            currentHour={currentHour}
            setCurrentHour={setCurrentHour}
            onStep={stepSimulation}
            onExplain={fetchExplainability}
            onReset={resetSimulation}
            loading={simLoading}
          />
        )}

        {/* Explain panel */}
        {activePanel === "explain" && explainData && (
          <ExplainPanel data={explainData} onClose={() => setActivePanel(simulation ? "simulation" : "fires")} />
        )}

        {/* Risk panel */}
        {activePanel === "risk" && riskData && (
          <RiskPanel data={riskData} onClose={clearRisk} onSimulate={(lat, lon) => startSimulation(lat, lon, 1.0)} />
        )}

        {/* Fire tips panel — shown after declaring a fire */}
        {showFireTips && fireTipsLocation && (
          <FireTipsPanel
            location={fireTipsLocation}
            onSimulate={() => { setShowFireTips(false); startSimulation(fireTipsLocation.lat, fireTipsLocation.lon, 1.0); }}
            onClose={() => setShowFireTips(false)}
          />
        )}
      </div>

      {/* About modal */}
      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}

      {/* Welcome guide */}
      {showWelcome && <WelcomeGuide onDismiss={dismissWelcome} />}
    </div>
  );
}
