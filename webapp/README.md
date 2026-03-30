# 🔥 FireSpreadNet — Web Application

Application web de surveillance et prédiction des feux de forêt, inspirée de [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/map/).

## Architecture

```
webapp/
├── backend/          # FastAPI (Python 3.11)
│   ├── main.py       # Point d'entrée FastAPI
│   ├── config.py     # Configuration & normalisation
│   ├── api/          # Routes REST
│   └── services/     # Logique métier (modèle, météo, FIRMS, simulation)
├── frontend/         # React 18 + Vite + Leaflet
│   ├── src/
│   │   ├── components/   # MapView, Sidebar, SimulationPanel, ExplainPanel...
│   │   ├── locales/      # FR / EN translations
│   │   └── services/     # Client API Axios
│   └── nginx.conf        # Reverse proxy (production)
├── docker-compose.yml
└── GCP_DEPLOY.md
```

## Fonctionnalités

- 🗺️ **Carte interactive** — Fond sombre (CartoDB Dark Matter), zoom, géolocation
- 🔥 **Feux actifs** — Données NASA FIRMS en temps réel (ou échantillon démo)
- 📍 **Déclarer un feu** — Clic sur la carte pour poser un point de départ
- ⏩ **Simulation** — Propagation heure par heure (jusqu'à 24h) via PI-CCA / U-Net / ConvLSTM
- 🧠 **Explicabilité** — Importance des variables (SHAP-based)
- 🌐 **Bilingue** — Français / English
- 🐳 **Docker** — Dockerfiles + docker-compose
- ☁️ **GCP** — Guide Cloud Run

---

## Lancement local (développement)

### Prérequis

- **Python 3.11+** (backend)
- **Node.js 18+** (frontend)
- Les modèles entraînés dans `saved_models/` (au moins un de : pi_cca, unet, convlstm)

### 1. Backend

```powershell
cd webapp/backend

# Créer un environnement virtuel
python -m venv venv
venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt

# (Optionnel) Configurer la clé FIRMS
copy .env.example .env
# Éditer .env et ajouter votre clé FIRMS_MAP_KEY

# Lancer le serveur
uvicorn main:app --reload --port 8000
```

Le backend est accessible sur **http://localhost:8000**
- Health check : http://localhost:8000/api/health
- Docs Swagger : http://localhost:8000/docs

### 2. Frontend

```powershell
cd webapp/frontend

# Installer les dépendances
npm install

# Lancer le serveur de développement
npm run dev
```

Le frontend est accessible sur **http://localhost:5173**

Le proxy Vite redirige automatiquement `/api/*` vers le backend sur le port 8000.

---

## Lancement avec Docker

```powershell
cd webapp

# Construire et lancer
docker compose up --build

# Accéder à l'application
# Frontend : http://localhost
# Backend  : http://localhost:8000
```

---

## Déploiement GCP

Voir [GCP_DEPLOY.md](GCP_DEPLOY.md) pour les instructions Cloud Run.

---

## API Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/fires/active` | Feux actifs (FIRMS) |
| POST | `/api/fires/declare` | Déclarer un feu |
| GET | `/api/fires/declared` | Liste des feux déclarés |
| POST | `/api/simulation/start` | Démarrer une simulation |
| POST | `/api/simulation/{id}/step` | Avancer la simulation |
| GET | `/api/simulation/{id}` | État de la simulation |
| GET | `/api/explainability/{sim_id}` | Explicabilité |

---

## Stack Technique

- **Backend** : FastAPI, PyTorch, NumPy, SciPy, scikit-learn, httpx
- **Frontend** : React 18, Vite, Leaflet, TailwindCSS, i18next, Recharts, Lucide React
- **APIs** : NASA FIRMS, Open-Meteo (météo), Open-Elevation (altitude)
- **Infra** : Docker, nginx, GCP Cloud Run
