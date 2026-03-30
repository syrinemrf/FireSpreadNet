# GCP Cloud Run deployment
# Deploy backend and frontend as separate Cloud Run services.
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#   gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com
#
# ─── Build & push images ────────────────────────────────────────────
#
# Backend:
#   cd webapp/backend
#   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/firespreadnet-backend .
#
# Frontend (build with production API URL):
#   cd webapp/frontend
#   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/firespreadnet-frontend .
#
# ─── Deploy services ────────────────────────────────────────────────
#
# Backend:
#   gcloud run deploy firespreadnet-backend \
#     --image gcr.io/YOUR_PROJECT_ID/firespreadnet-backend \
#     --platform managed \
#     --region europe-west1 \
#     --memory 2Gi \
#     --cpu 2 \
#     --timeout 120 \
#     --set-env-vars "FIRMS_MAP_KEY=your_key_here" \
#     --allow-unauthenticated
#
# Frontend:
#   gcloud run deploy firespreadnet-frontend \
#     --image gcr.io/YOUR_PROJECT_ID/firespreadnet-frontend \
#     --platform managed \
#     --region europe-west1 \
#     --memory 256Mi \
#     --allow-unauthenticated
#
# Note: For the frontend to reach the backend in production, update
# the nginx.conf proxy_pass to the backend Cloud Run URL, or use
# a load balancer / API Gateway.
