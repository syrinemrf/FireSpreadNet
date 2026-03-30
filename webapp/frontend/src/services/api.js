import axios from "axios";

const api = axios.create({
  baseURL: "",       // Uses Vite proxy in dev, nginx proxy in prod
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    console.error("[API]", err.response?.data || err.message);
    return Promise.reject(err);
  }
);

export default api;
