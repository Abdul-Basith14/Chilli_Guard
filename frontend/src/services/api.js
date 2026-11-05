/**
 * ChilliGuard API client.
 *
 * All backend communication goes through this module.
 * Base URL is read from the VITE_API_BASE_URL environment variable,
 * falling back to the Vite dev proxy path ('/api').
 *
 * No credentials, tokens, or secrets are included in frontend code.
 */

import axios from 'axios'

// During development the Vite proxy forwards /api → http://127.0.0.1:8000
// In production set VITE_API_BASE_URL to the deployed backend URL.
const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000, // 30 s — inference can take a moment on CPU
  headers: { Accept: 'application/json' },
})

// ── Health ────────────────────────────────────────────────────────────────────

/**
 * GET /api/health
 * @returns {{ status, environment, timestamp, version }}
 */
export async function fetchHealth() {
  const { data } = await apiClient.get('/api/health')
  return data
}

// ── Model status ──────────────────────────────────────────────────────────────

/**
 * GET /api/model/status
 * @returns {{ status, model_name, checkpoint_path, num_classes, message }}
 */
export async function fetchModelStatus() {
  const { data } = await apiClient.get('/api/model/status')
  return data
}

/**
 * GET /api/model/metrics
 * @returns {{ status, exp_a, exp_b, audit, test_results }}
 */
export async function fetchModelMetrics() {
  const { data } = await apiClient.get('/api/model/metrics')
  return data
}

// ── Dataset Inspection ────────────────────────────────────────────────────────

/**
 * GET /api/dataset/status
 */
export async function fetchDatasetStatus() {
  const { data } = await apiClient.get('/api/dataset/status')
  return data
}

/**
 * POST /api/dataset/inspect
 */
export async function triggerDatasetInspection() {
  const { data } = await apiClient.post('/api/dataset/inspect')
  return data
}

/**
 * GET /api/dataset/report
 */
export async function fetchDatasetReport() {
  const { data } = await apiClient.get('/api/dataset/report')
  return data
}

/**
 * Get chart URL for dataset visualizations
 */
export function getDatasetChartUrl(chartName) {
  return `${BASE_URL}/api/dataset/charts/${chartName}`
}

// ── Unified Prediction ────────────────────────────────────────────────────────

/**
 * POST /api/predict
 * Unified inference endpoint for binary image, 19-class image, or FTIR spectral samples.
 * @param {File} file Uploaded file object
 * @param {'binary' | 'multiclass' | 'ftir'} mode Prediction mode
 * @returns {Promise<Object>}
 */
export async function predictSample(file, mode = 'binary') {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('mode', mode)

  const { data } = await apiClient.post('/api/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

/** Alias for binary image prediction */
export const predictImage = (file) => predictSample(file, 'binary')

export default apiClient


