#!/usr/bin/env python3
"""
src/simulation/rothermel.py — Rothermel Fire Spread Model (Vectorised)
======================================================================
Implements the Rothermel (1972) surface fire spread equations in a
grid-compatible, fully vectorised form suitable for:
  • Classical cellular-automaton simulations (ground truth generation)
  • Differentiable physics branch inside the PI-CCA model (PyTorch)

Key equations
-------------
Rate of spread  R = (I_R · ξ · (1 + φ_w + φ_s)) / (ρ_b · ε · Q_ig)

Where:
  I_R   = reaction intensity  (kW m⁻²)
  ξ     = propagating flux ratio
  φ_w   = wind coefficient
  φ_s   = slope factor
  ρ_b   = ovendry bulk density  (kg m⁻³)
  ε     = effective heating number
  Q_ig  = heat of preignition  (kJ kg⁻¹)

References
----------
  Rothermel, R.C. (1972). A mathematical model for predicting fire spread
      in wildland fuels.  USDA Forest Service Research Paper INT-115.
  Andrews, P.L. (2018). The Rothermel surface fire spread model and
      associated developments: A comprehensive explanation. Gen. Tech. Rep.
      RMRS-GTR-371.
"""

import numpy as np
from typing import Tuple, Optional


# ── Physical constants ─────────────────────────────────────────
SIGMA_SB = 5.670374419e-8   # Stefan-Boltzmann  (W m⁻² K⁻⁴)
HEAT_CONTENT = 18_622.0     # Low heat of combustion  (kJ kg⁻¹)
PARTICLE_DENSITY = 513.0    # Ovendry particle density (kg m⁻³)


class RothermelModel:
    """Vectorised Rothermel surface fire spread calculator.

    Works on 2-D numpy grids so the entire landscape is processed
    in a single call (no cell-by-cell loops).
    """

    def __init__(self, cell_size: float = 250.0, timestep: float = 3600.0):
        """
        Parameters
        ----------
        cell_size : float
            Grid cell edge length in metres.
        timestep : float
            Simulation time step in seconds.
        """
        self.cell_size = cell_size
        self.timestep = timestep

    # ─── core spread rate ──────────────────────────────────────
    def rate_of_spread(
        self,
        fuel_load: np.ndarray,
        fuel_depth: np.ndarray,
        fuel_moisture: np.ndarray,
        moisture_ext: np.ndarray,
        heat_content: np.ndarray,
        wind_speed: np.ndarray,
        slope: np.ndarray,
    ) -> np.ndarray:
        """Compute Rothermel rate of spread R (m s⁻¹) for every cell.

        All inputs are 2-D arrays of shape (H, W).
        """
        # Derived quantities
        fuel_load = np.clip(fuel_load, 1e-6, None)
        fuel_depth = np.clip(fuel_depth, 1e-3, None)

        # Bulk density
        rho_b = fuel_load / fuel_depth                           # kg m⁻³

        # Packing ratio
        beta = rho_b / PARTICLE_DENSITY
        beta = np.clip(beta, 1e-8, 0.5)

        # Optimal packing ratio  (Rothermel eq. 37)
        sigma = 60.0   # surface-area-to-volume ratio ft⁻¹ (representative)
        beta_op = 3.348 * sigma ** (-0.8189)

        # Relative packing ratio
        beta_rel = beta / beta_op

        # Reaction intensity  I_R  (Rothermel eq. 27)
        gamma_max = (sigma ** 1.5) / (495.0 + 0.0594 * sigma ** 1.5)
        A = 133.0 * sigma ** (-0.7913)
        gamma = gamma_max * (beta_rel ** A) * np.exp(A * (1.0 - beta_rel))

        # Moisture damping  η_M
        rm = np.clip(fuel_moisture / np.clip(moisture_ext, 1e-6, None), 0, 1)
        eta_M = 1.0 - 2.59 * rm + 5.11 * rm**2 - 3.52 * rm**3
        eta_M = np.clip(eta_M, 0, 1)

        I_R = gamma * heat_content * fuel_load * eta_M           # kW m⁻²

        # Propagating flux ratio  ξ  (Rothermel eq. 42)
        xi = np.exp((0.792 + 0.681 * sigma**0.5) * (beta + 0.1)) / \
             (192.0 + 0.2595 * sigma)
        xi = np.clip(xi, 1e-6, 1.0)

        # Wind coefficient  φ_w (Rothermel eq. 47)
        C_wind = 7.47 * np.exp(-0.133 * sigma ** 0.55)
        B_wind = 0.02526 * sigma ** 0.54
        E_wind = 0.715 * np.exp(-3.59e-4 * sigma)
        phi_w = C_wind * (wind_speed ** B_wind) * (beta_rel ** (-E_wind))

        # Slope factor  φ_s (Rothermel eq. 51)
        tan_slope = np.tan(np.deg2rad(np.clip(slope, 0, 80)))
        phi_s = 5.275 * beta ** (-0.3) * tan_slope ** 2

        # Effective heating number  ε
        epsilon = np.exp(-138.0 / sigma)

        # Heat of preignition  Q_ig (Rothermel eq. 78)
        Q_ig = 250.0 + 1116.0 * fuel_moisture                   # kJ kg⁻¹

        # Rate of spread  R  (Rothermel eq. 52)
        numerator = I_R * xi * (1.0 + phi_w + phi_s)
        denominator = rho_b * epsilon * Q_ig
        denominator = np.clip(denominator, 1e-6, None)
        R = numerator / denominator                              # m s⁻¹
        R = np.clip(R, 0, None)

        return R

    # ─── directional spread (wind + slope) ─────────────────────
    def directional_spread_probability(
        self,
        R: np.ndarray,
        wind_speed: np.ndarray,
        wind_direction: np.ndarray,
        slope: np.ndarray,
        aspect: np.ndarray,
    ) -> np.ndarray:
        """Compute 8-directional spread probabilities from a burning cell.

        Returns shape (8, H, W) with spread probability along each of
        the 8 Moore-neighbourhood directions.

        Directions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        """
        # Direction angles (mathematical, CCW from East)
        dir_angles = np.array([90, 45, 0, 315, 270, 225, 180, 135], dtype=np.float64)

        H, W = R.shape
        probs = np.zeros((8, H, W), dtype=np.float64)

        # Convert wind direction from meteorological (CW from N) to math
        wind_math = (270.0 - wind_direction) % 360.0

        for i, d_angle in enumerate(dir_angles):
            # Wind alignment factor
            angle_diff = np.deg2rad(d_angle - wind_math)
            wind_factor = np.clip(np.cos(angle_diff), -1, 1)
            wind_effect = 1.0 + 0.5 * wind_factor * (wind_speed / 10.0)

            # Upslope factor
            aspect_diff = np.deg2rad(d_angle - aspect)
            slope_rad = np.deg2rad(slope)
            upslope = np.sin(slope_rad) * np.cos(aspect_diff)
            slope_effect = 1.0 + 0.3 * upslope

            # Distance factor (diagonal = √2)
            dist = self.cell_size * (np.sqrt(2) if i % 2 == 1 else 1.0)

            # Probability that fire reaches neighbour in one timestep
            spread_dist = R * self.timestep * wind_effect * slope_effect
            p = np.clip(spread_dist / dist, 0, 1)
            probs[i] = p

        return probs


class FirePropagationEngine:
    """Run a full fire propagation simulation on a grid.

    States
    ------
    0 = unburned,  1 = burning,  2 = burned-out

    The engine uses the Rothermel model to compute spread probabilities
    and a stochastic cellular automaton to advance the fire front.
    """

    UNBURNED = 0
    BURNING = 1
    BURNED = 2

    def __init__(
        self,
        grid_size: int = 64,
        cell_size: float = 250.0,
        timestep: float = 3600.0,
        burnout_steps: int = 3,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.timestep = timestep
        self.burnout_steps = burnout_steps
        self.rng = np.random.default_rng(seed)
        self.rothermel = RothermelModel(cell_size, timestep)

    # ─── terrain generation (synthetic Mediterranean) ──────────
    def generate_terrain(self) -> dict:
        """Create a synthetic but realistic Mediterranean terrain grid.

        Returns dict with keys: dem, slope, aspect, fuel_type,
        fuel_moisture, canopy_cover — each shaped (H, W).
        """
        N = self.grid_size
        rng = self.rng

        # DEM via superposition of Perlin-like smooth noise
        dem = self._smooth_noise(N, octaves=4, scale=0.05) * 1500 + 300
        dem = np.clip(dem, 0, 2500)

        # Slope & aspect from DEM
        dy, dx = np.gradient(dem, self.cell_size)
        slope = np.rad2deg(np.arctan(np.sqrt(dx**2 + dy**2)))
        aspect = np.rad2deg(np.arctan2(-dx, dy)) % 360

        # Fuel type — altitude / noise-based
        fuel_noise = self._smooth_noise(N, octaves=2, scale=0.08)
        fuel_type = np.zeros((N, N), dtype=np.int32)
        fuel_type[fuel_noise < -0.3] = 1   # short grass (low areas)
        fuel_type[(fuel_noise >= -0.3) & (fuel_noise < 0.0)] = 4   # chaparral
        fuel_type[(fuel_noise >= 0.0) & (fuel_noise < 0.3)] = 14   # maquis (Mediterranean)
        fuel_type[(fuel_noise >= 0.3) & (fuel_noise < 0.6)] = 15   # garrigue
        fuel_type[fuel_noise >= 0.6] = 8   # closed timber (mountains)
        fuel_type[dem > 2000] = 0   # no fuel (bare rock)

        # Add water bodies / no-fuel patches
        water_mask = self._smooth_noise(N, octaves=1, scale=0.03) > 0.85
        fuel_type[water_mask] = 0

        # Fuel moisture — depends on altitude & randomness
        fuel_moisture = 0.06 + 0.15 * (dem / 2500) + rng.uniform(-0.02, 0.02, (N, N))
        fuel_moisture = np.clip(fuel_moisture, 0.03, 0.30)

        # Canopy cover
        canopy = self._smooth_noise(N, octaves=2, scale=0.06) * 0.5 + 0.3
        canopy[fuel_type == 0] = 0
        canopy[fuel_type == 1] = np.clip(canopy[fuel_type == 1], 0, 0.15)
        canopy = np.clip(canopy, 0, 1)

        return {
            "dem": dem.astype(np.float32),
            "slope": slope.astype(np.float32),
            "aspect": aspect.astype(np.float32),
            "fuel_type": fuel_type,
            "fuel_moisture": fuel_moisture.astype(np.float32),
            "canopy_cover": canopy.astype(np.float32),
        }

    # ─── weather generation ────────────────────────────────────
    def generate_weather(self, n_steps: int) -> dict:
        """Generate weather conditions for *n_steps* hours.

        Returns dict with arrays of shape (n_steps,) — spatially
        uniform but temporally varying.
        """
        rng = self.rng
        # Base wind — slowly varying
        base_ws = rng.uniform(1.0, 12.0)
        wind_speed = base_ws + np.cumsum(rng.normal(0, 0.3, n_steps))
        wind_speed = np.clip(wind_speed, 0.5, 20.0).astype(np.float32)

        base_wd = rng.uniform(0, 360)
        wind_dir = (base_wd + np.cumsum(rng.normal(0, 5, n_steps))) % 360
        wind_dir = wind_dir.astype(np.float32)

        # Temperature — diurnal cycle approximation
        base_temp = rng.uniform(25, 42)
        hours = np.arange(n_steps)
        temp = base_temp + 5 * np.sin(2 * np.pi * (hours - 6) / 24)
        temp = temp.astype(np.float32)

        # Humidity — inversely correlated with temperature
        hum = 0.65 - 0.4 * (temp - temp.min()) / (temp.max() - temp.min() + 1e-6)
        hum = np.clip(hum + rng.normal(0, 0.03, n_steps), 0.1, 0.9).astype(np.float32)

        return {
            "wind_speed": wind_speed,
            "wind_direction": wind_dir,
            "temperature": temp,
            "humidity": hum,
        }

    # ─── run simulation ────────────────────────────────────────
    def simulate(
        self,
        terrain: dict,
        weather: dict,
        ignition_points: list,
        max_hours: Optional[int] = None,
    ) -> dict:
        """Run the fire propagation simulation.

        Parameters
        ----------
        terrain : dict of (H, W) arrays
        weather : dict of (T,) arrays
        ignition_points : list of (row, col) tuples
        max_hours : int or None

        Returns
        -------
        dict with:
          fire_states : (T+1, H, W) int array  (0/1/2)
          time_burning : (T+1, H, W) float array
          burned_area_per_step : list of float  (km²)
        """
        # Fuel model properties (local copy — no config dependency)
        FUEL_MODELS = {
            0:  {"fuel_load": 0.0,  "SAV": 0,    "depth": 0.0, "Mx": 0.0,  "desc": "No fuel"},
            1:  {"fuel_load": 0.74, "SAV": 3500, "depth": 0.30, "Mx": 0.12, "desc": "Short grass"},
            2:  {"fuel_load": 2.00, "SAV": 3000, "depth": 0.30, "Mx": 0.15, "desc": "Timber grass"},
            3:  {"fuel_load": 3.01, "SAV": 1500, "depth": 0.76, "Mx": 0.25, "desc": "Tall grass"},
            4:  {"fuel_load": 5.01, "SAV": 2000, "depth": 1.83, "Mx": 0.20, "desc": "Chaparral"},
            5:  {"fuel_load": 1.00, "SAV": 2000, "depth": 0.61, "Mx": 0.20, "desc": "Brush"},
            6:  {"fuel_load": 1.50, "SAV": 1750, "depth": 0.76, "Mx": 0.25, "desc": "Dormant brush"},
            7:  {"fuel_load": 1.13, "SAV": 1550, "depth": 0.76, "Mx": 0.40, "desc": "Southern rough"},
            8:  {"fuel_load": 1.50, "SAV": 2000, "depth": 0.06, "Mx": 0.30, "desc": "Closed timber"},
            9:  {"fuel_load": 0.41, "SAV": 2500, "depth": 0.06, "Mx": 0.25, "desc": "Hardwood litter"},
            10: {"fuel_load": 3.01, "SAV": 2000, "depth": 0.30, "Mx": 0.25, "desc": "Timber understory"},
            11: {"fuel_load": 1.50, "SAV": 1500, "depth": 0.30, "Mx": 0.15, "desc": "Light logging"},
            12: {"fuel_load": 4.01, "SAV": 1750, "depth": 0.70, "Mx": 0.20, "desc": "Medium logging"},
            13: {"fuel_load": 7.01, "SAV": 1750, "depth": 0.91, "Mx": 0.25, "desc": "Heavy logging"},
            14: {"fuel_load": 4.50, "SAV": 1800, "depth": 1.20, "Mx": 0.30, "desc": "Maquis"},
            15: {"fuel_load": 2.50, "SAV": 2200, "depth": 0.50, "Mx": 0.15, "desc": "Garrigue"},
        }

        N = self.grid_size
        max_h = max_hours or len(weather["wind_speed"])
        n_steps = min(max_h, len(weather["wind_speed"]))

        # State grids
        state = np.zeros((N, N), dtype=np.int32)
        time_burning = np.zeros((N, N), dtype=np.float32)

        # Ignite
        for r, c in ignition_points:
            if 0 <= r < N and 0 <= c < N:
                state[r, c] = self.BURNING

        # Fuel properties grids
        ft = terrain["fuel_type"]
        fuel_load = np.zeros((N, N), dtype=np.float32)
        fuel_depth = np.zeros((N, N), dtype=np.float32)
        moisture_ext = np.zeros((N, N), dtype=np.float32)
        heat_arr = np.zeros((N, N), dtype=np.float32)

        for fid, props in FUEL_MODELS.items():
            mask = ft == fid
            fuel_load[mask] = props["load"]
            fuel_depth[mask] = props["depth"]
            moisture_ext[mask] = props["moisture_ext"]
            heat_arr[mask] = props["heat"]

        # Storage
        fire_states = [state.copy()]
        time_burnings = [time_burning.copy()]
        burned_areas = []

        for t in range(n_steps):
            ws = np.full((N, N), weather["wind_speed"][t], dtype=np.float32)
            wd = np.full((N, N), weather["wind_direction"][t], dtype=np.float32)

            # Adjust fuel moisture with humidity
            fm = terrain["fuel_moisture"] * (1.0 + 0.5 * (weather["humidity"][t] - 0.4))
            fm = np.clip(fm, 0.03, 0.35)

            # Rate of spread
            R = self.rothermel.rate_of_spread(
                fuel_load, fuel_depth, fm, moisture_ext, heat_arr, ws, terrain["slope"]
            )

            # Directional probabilities
            probs = self.rothermel.directional_spread_probability(
                R, ws, wd, terrain["slope"], terrain["aspect"]
            )

            # New state
            new_state = state.copy()

            # Offsets for 8 neighbours (N, NE, E, SE, S, SW, W, NW)
            offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                       (1, 0), (1, -1), (0, -1), (-1, -1)]

            # For each burning cell, try to ignite neighbours
            burning_mask = (state == self.BURNING)

            for di, (dr, dc) in enumerate(offsets):
                # Shift the burning mask
                shifted_burn = np.zeros_like(burning_mask)
                src_r = slice(max(0, -dr), N + min(0, -dr))
                dst_r = slice(max(0, dr), N + min(0, dr))
                src_c = slice(max(0, -dc), N + min(0, -dc))
                dst_c = slice(max(0, dc), N + min(0, dc))
                shifted_burn[src_r, src_c] = burning_mask[dst_r, dst_c]

                # Candidate cells: unburned and has a burning neighbour
                candidates = (state == self.UNBURNED) & shifted_burn

                # Stochastic ignition
                rand = self.rng.random((N, N))
                ignite = candidates & (rand < probs[di])
                new_state[ignite] = self.BURNING

            # Burnout: cells that have been burning too long
            time_burning[burning_mask] += 1
            burnout = (state == self.BURNING) & (time_burning >= self.burnout_steps)
            new_state[burnout] = self.BURNED

            # Cannot burn no-fuel cells
            new_state[ft == 0] = state[ft == 0]

            state = new_state
            fire_states.append(state.copy())
            time_burnings.append(time_burning.copy())
            burned = np.sum((state == self.BURNING) | (state == self.BURNED))
            burned_areas.append(burned * (self.cell_size ** 2) / 1e6)  # km²

        return {
            "fire_states": np.stack(fire_states),       # (T+1, H, W)
            "time_burning": np.stack(time_burnings),     # (T+1, H, W)
            "burned_area_km2": burned_areas,
        }

    # ─── helper: smooth noise ──────────────────────────────────
    def _smooth_noise(self, size: int, octaves: int = 4, scale: float = 0.05) -> np.ndarray:
        """Generate smooth 2-D noise via superposition of random fields."""
        result = np.zeros((size, size))
        for o in range(octaves):
            freq = scale * (2 ** o)
            amp = 1.0 / (2 ** o)
            # Low-resolution random field upsampled with interpolation
            n = max(3, int(size * freq) + 1)
            small = self.rng.standard_normal((n, n))
            # Bilinear upsample
            from scipy.ndimage import zoom
            large = zoom(small, size / n, order=1)[:size, :size]
            result += amp * large
        # Normalise to [-1, 1]
        result = (result - result.mean()) / (result.std() + 1e-8)
        return result
