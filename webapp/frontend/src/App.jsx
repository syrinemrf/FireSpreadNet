import { useState, useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import MapView from "./components/MapView";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import SimulationPanel from "./components/SimulationPanel";
import ExplainPanel from "./components/ExplainPanel";
import AboutModal from "./components/AboutModal";
import api from "./services/api";
import useGeolocation from "./hooks/useGeolocation";

export default function App() {
  const { t } = useTranslation();
  const { position: geoPos, locate } = useGeolocation();

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
  const [activePanel, setActivePanel] = useState("fires");
  const [showAbout, setShowAbout] = useState(false);
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
        // Direct simulation mode — click map to start simulation immediately
        startSimulation(latlng.lat, latlng.lng, 1.0);
        return;
      }
      if (declaring) {
        setPendingDeclare({ lat: latlng.lat, lon: latlng.lng, radius_km: 1.0 });
        return;
      }
    },
    [declaring, simulating]
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
      setDeclaredFires((prev) => [...prev, { ...resp.data.fire, lat: pendingDeclare.lat, lon: pendingDeclare.lon }]);
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

  const resetSimulation = useCallback(() => {
    setSimulation(null);
    setSimGeoJson(null);
    setCurrentHour(0);
    setExplainData(null);
    setActivePanel("fires");
  }, []);

  /* ── geolocation ───────────────────────── */
  const handleLocate = useCallback(() => {
    locate();
    if (geoPos) {
      setMapCenter([geoPos.lat, geoPos.lon]);
      setMapZoom(12);
    } else {
      navigator.geolocation?.getCurrentPosition(
        (pos) => {
          setMapCenter([pos.coords.latitude, pos.coords.longitude]);
          setMapZoom(12);
        },
        () => {},
        { enableHighAccuracy: true, timeout: 10000 }
      );
    }
  }, [locate, geoPos]);

  /* ── search location ───────────────────── */
  const handleSearch = useCallback(async (query) => {
    try {
      const resp = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`
      );
      const data = await resp.json();
      if (data.length > 0) {
        setMapCenter([parseFloat(data[0].lat), parseFloat(data[0].lon)]);
        setMapZoom(10);
      }
    } catch {
      /* ignore geocoding errors */
    }
  }, []);

  /* ── active interaction mode label ─────── */
  const interactionMode = simulating ? "simulation" : declaring ? "declare" : null;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-dark-900">
      {/* Mobile menu toggle */}
      <button
        className="fixed top-3 left-3 z-[1100] sm:hidden w-10 h-10 bg-dark-700 border border-dark-400 rounded-lg flex items-center justify-center text-white shadow-lg"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        {sidebarOpen ? "✕" : "☰"}
      </button>

      {/* Sidebar */}
      <div className={`${sidebarOpen ? "translate-x-0" : "-translate-x-full"} sm:translate-x-0 fixed sm:relative z-[1050] transition-transform duration-200`}>
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
          fireCount={fires.length}
          hasSimulation={!!simulation}
        />
      </div>

      {/* Mobile overlay backdrop */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-[1040] sm:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Main area */}
      <div className="flex-1 flex flex-col relative">
        <TopBar onSearch={handleSearch} />

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
        />

        {/* Loading indicator */}
        {loading && fires.length === 0 && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 z-[1200] bg-dark-700/95 backdrop-blur text-white text-xs sm:text-sm px-4 py-2 rounded-xl shadow-lg border border-dark-400/50 flex items-center gap-2">
            <span className="inline-block w-3 h-3 border-2 border-fire-400 border-t-transparent rounded-full animate-spin"></span>
            Loading active fires...
          </div>
        )}

        {/* Simulation loading overlay */}
        {simLoading && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 z-[1200] bg-dark-700/95 backdrop-blur text-white text-xs sm:text-sm px-4 py-2 rounded-xl shadow-lg border border-fire-500/50 flex items-center gap-2">
            <span className="inline-block w-3 h-3 border-2 border-fire-400 border-t-transparent rounded-full animate-spin"></span>
            🔥 Simulating fire spread...
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
            <span className="text-green-400 text-xs sm:text-sm font-medium">🔥 Click on the map to start a fire simulation</span>
            <button onClick={() => setSimulating(false)} className="text-gray-400 hover:text-white text-sm">
              ✕
            </button>
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
                <button onClick={() => { confirmDeclare(); startSimFromPending(); }} className="px-3 py-1 bg-fire-600 text-white rounded-lg text-xs hover:bg-fire-500 transition">
                  ▶ {t("declare.start_simulation")}
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
      </div>

      {/* About modal */}
      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
    </div>
  );
}
