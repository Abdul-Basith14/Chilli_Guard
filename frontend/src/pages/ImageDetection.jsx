/**
 * Unified Detection Page — ChilliGuard Adulteration Detection System.
 *
 * Supports three distinct classification modes:
 * 1. Binary Image Classification (Pure vs. Adulterated)
 * 2. 19-Class Image Classification (Adulterant type & concentration)
 * 3. ATR-FTIR Spectral Classification (Pure vs. Sudan IV + % w/w estimation)
 */

import { useState } from 'react'
import SampleUploader from '../components/SampleUploader.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import { predictSample } from '../services/api.js'
import { getClassDetails } from '../utils/classDescriptions.js'

// ── Mode metadata ─────────────────────────────────────────────────────────────

const MODES = [
  {
    id: 'binary',
    title: 'Binary Image Detection',
    subtitle: 'Pure vs. Adulterated Chilli Powder',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="w-4 h-4">
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <circle cx="8.5" cy="8.5" r="1.5" />
        <path d="M21 15l-5-5L5 21" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    acceptDesc: 'JPEG, PNG, WebP, BMP, TIFF · Max 10 MB',
    modelBadge: 'ResNet18 Binary Classifier (82.0% test accuracy)',
    description: 'Classifies RGB images of red chilli powder as Pure or Adulterated using deep feature extraction.',
  },
  {
    id: 'multiclass',
    title: '19-Class Image Detection',
    subtitle: 'Adulterant Type & Concentration Class',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="w-4 h-4">
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </svg>
    ),
    acceptDesc: 'JPEG, PNG, WebP, BMP, TIFF · Max 10 MB',
    modelBadge: 'ResNet18 19-Class Fine-Tuned (96.3% top-5 accuracy)',
    description: 'Identifies specific adulterant mixtures and concentrations across 19 trained fine-grained categories with top-5 breakdown.',
  },
  {
    id: 'ftir',
    title: 'ATR-FTIR Spectral Detection',
    subtitle: 'Pure vs. Sudan IV + Concentration %',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="w-4 h-4">
        <path d="M2 12h3l2-6 4 12 4-10 2 4h5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    acceptDesc: 'CSV, XLSX, XLS, NPY Spectral Data · Max 10 MB',
    modelBadge: 'SNV + SG 1st Deriv + Linear SVM (100% held-out test accuracy)',
    description: 'Analyses mid-infrared absorption spectra (700-3700 cm⁻¹) to detect Sudan IV dye adulteration and estimate % w/w concentration.',
  },
]

// ── Result states ─────────────────────────────────────────────────────────────

function EmptyState({ activeMode }) {
  const currentMode = MODES.find((m) => m.id === activeMode)
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3 py-12 text-center">
      <div className="w-12 h-12 rounded-full bg-surface-700/80 border border-surface-600 flex items-center justify-center">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="w-6 h-6 text-ink-400">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 8v4M12 16h.01" strokeLinecap="round" />
        </svg>
      </div>
      <p className="text-sm text-ink-300 font-medium">No sample selected</p>
      <p className="text-xs text-ink-500 max-w-xs leading-relaxed">
        Upload a file on the left using <span className="text-olive-400 font-medium">{currentMode?.title}</span> mode to run prediction.
      </p>
    </div>
  )
}

function LoadingState({ mode }) {
  const modeTitle = MODES.find((m) => m.id === mode)?.title || 'Sample'
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 py-12">
      <div className="w-10 h-10 border-2 border-surface-600 border-t-olive-400 rounded-full animate-spin" aria-hidden="true" />
      <p className="text-sm text-ink-300 animate-pulse-olive font-medium">Running {modeTitle}…</p>
      <p className="text-xs text-ink-500">Processing sample with model pipeline</p>
    </div>
  )
}

function ErrorState({ message }) {
  return (
    <div className="flex flex-col items-center gap-3 py-12 text-center">
      <div className="w-12 h-12 rounded-full bg-red-950/40 border border-red-800/40 flex items-center justify-center">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="w-6 h-6 text-red-400">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 8v4M12 16h.01" strokeLinecap="round" />
        </svg>
      </div>
      <p className="text-sm font-semibold text-red-400">Prediction Failed</p>
      <p className="text-xs text-ink-400 max-w-xs leading-relaxed">{message}</p>
    </div>
  )
}

function ModelNotReadyState({ message }) {
  return (
    <div className="flex flex-col items-center gap-4 py-12 text-center">
      <div className="w-14 h-14 rounded-full bg-surface-700/60 border border-surface-600 flex items-center justify-center">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="w-7 h-7 text-ink-400">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 6v6l4 2" strokeLinecap="round" />
        </svg>
      </div>
      <div>
        <StatusBadge status="not_trained" />
        <p className="text-sm font-medium text-ink-200 mt-3">Model Not Available</p>
        <p className="text-xs text-ink-400 max-w-sm mt-1.5 leading-relaxed">{message}</p>
      </div>
    </div>
  )
}

function PredictionReadyState({ result, mode }) {
  const isPure = result.predicted_class === 'Pure'
  const modeMeta = MODES.find((m) => m.id === mode)

  return (
    <div className="space-y-5 animate-slide-up">
      {/* Model Mode Tag */}
      <div className="flex items-center justify-between text-[11px] pb-2 border-b border-surface-700/50">
        <span className="text-ink-400 font-medium flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-olive-400 inline-block" />
          {modeMeta?.title}
        </span>
        <span className="text-olive-400 font-mono text-[10px]">{modeMeta?.modelBadge}</span>
      </div>

      {/* Main Verdict Card */}
      <div
        className={`
          rounded-xl p-4 border flex items-center gap-4
          ${
            isPure
              ? 'bg-olive-950/40 border-olive-700/40'
              : 'bg-red-950/35 border-red-900/40'
          }
        `}
      >
        <div
          className={`
            w-12 h-12 rounded-xl flex items-center justify-center shrink-0 text-xl
            ${isPure ? 'bg-olive-900/50 text-olive-400' : 'bg-red-900/50 text-red-400'}
          `}
        >
          {isPure ? '✓' : '⚠'}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-ink-400 font-medium uppercase tracking-wider">Classification Verdict</p>
          <p className="text-lg font-bold text-ink-100 truncate mt-0.5">{result.predicted_class}</p>
          {result.confidence != null && (
            <p className="text-xs text-ink-300 mt-0.5 font-mono">
              Confidence Score: <span className="font-semibold text-ink-100">{(result.confidence * 100).toFixed(1)}%</span>
            </p>
          )}
        </div>
      </div>

      {/* FTIR Concentration Estimation Card */}
      {result.estimated_concentration_percent != null && (
        <div className="bg-surface-800/80 border border-amber-500/30 rounded-xl p-4 flex items-center justify-between">
          <div>
            <p className="text-[11px] font-medium text-amber-400 uppercase tracking-wider">Adulterant Concentration</p>
            <p className="text-xs text-ink-400 mt-0.5">Estimated Sudan IV dye content (% w/w)</p>
          </div>
          <div className="text-right">
            <span className="text-xl font-bold font-mono text-amber-300">{result.estimated_concentration_percent}%</span>
            <span className="text-xs text-ink-500 block">w/w</span>
          </div>
        </div>
      )}

      {/* Top-5 Predictions for Multiclass Mode */}
      {result.top5_predictions && result.top5_predictions.length > 0 && (
        <div className="space-y-2.5">
          <p className="text-xs font-semibold text-ink-300 uppercase tracking-wider">Top-5 Predicted Classes</p>
          <div className="space-y-2">
            {result.top5_predictions.map((item, idx) => (
              <div key={item.class_name} className="bg-surface-800/60 rounded-lg p-2.5 border border-surface-700/40">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-ink-200 font-medium">
                    <span className="text-ink-500 font-mono mr-1.5">#{idx + 1}</span>
                    {item.class_name}
                  </span>
                  <span className="text-ink-400 font-mono">{(item.probability * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-surface-900 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-olive-500 rounded-full transition-all duration-500"
                    style={{ width: `${(item.probability * 100).toFixed(1)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Class Probabilities Bar Chart (Binary or FTIR mode) */}
      {result.class_probabilities && !result.top5_predictions && (
        <div className="space-y-2.5">
          <p className="text-xs font-semibold text-ink-300 uppercase tracking-wider">Class Probabilities</p>
          <div className="space-y-2">
            {Object.entries(result.class_probabilities).map(([cls, prob]) => (
              <div key={cls} className="bg-surface-800/60 rounded-lg p-2.5 border border-surface-700/40">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-ink-200 font-medium">{cls}</span>
                  <span className="text-ink-400 font-mono">{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-surface-900 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      cls === 'Pure' ? 'bg-olive-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${(prob * 100).toFixed(1)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* File Details */}
      {result.filename && (
        <div className="text-[10px] text-ink-500 flex items-center justify-between pt-2 border-t border-surface-700/50">
          <span>Uploaded sample: <strong className="text-ink-300 font-normal">{result.filename}</strong></span>
          {result.file_size_bytes && <span>{(result.file_size_bytes / 1024).toFixed(1)} KB</span>}
        </div>
      )}

      {/* Scientific Disclaimer & Model Limitations */}
      <div className="bg-surface-900/90 border border-surface-700/60 rounded-lg p-3.5 space-y-1.5">
        <p className="text-[10px] font-semibold text-olive-400 uppercase tracking-wider flex items-center gap-1">
          <span>ℹ</span> Model Context & Disclaimers
        </p>
        <ul className="text-[11px] text-ink-400 space-y-1 list-disc list-inside leading-relaxed">
          {mode === 'binary' && (
            <li>Binary image model predicts binary adulteration state from visual texture.</li>
          )}
          {mode === 'multiclass' && (
            <li>19-class model predicts among its trained fine-grained adulterant categories.</li>
          )}
          {mode === 'ftir' && (
            <li>
              FTIR model is trained specifically on Pure vs. Sudan IV dye spectra. 100% test accuracy is evaluated on the held-out group-split test set (N=21), reflecting high ATR-FTIR mid-IR diagnostic sensitivity.
            </li>
          )}
          <li>Predictions do not constitute definitive chemical verification.</li>
        </ul>
      </div>
    </div>
  )
}

// ── Main Page Component ───────────────────────────────────────────────────────

export default function ImageDetection() {
  const [activeMode, setActiveMode]     = useState('binary')
  const [selectedFile, setSelectedFile] = useState(null)
  const [result, setResult]             = useState(null)
  const [loading, setLoading]           = useState(false)
  const [apiError, setApiError]         = useState(null)

  const activeMeta = MODES.find((m) => m.id === activeMode)

  const handleModeChange = (modeId) => {
    if (modeId === activeMode) return
    setActiveMode(modeId)
    setSelectedFile(null)
    setResult(null)
    setApiError(null)
  }

  const handleFileSelected = (file) => {
    setSelectedFile(file)
    setResult(null)
    setApiError(null)
  }

  const handleSubmit = async () => {
    if (!selectedFile) return
    setLoading(true)
    setResult(null)
    setApiError(null)

    try {
      const data = await predictSample(selectedFile, activeMode)
      setResult(data)
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        'Could not reach the ChilliGuard backend server. Please verify backend is running on http://127.0.0.1:8000.'
      setApiError(msg)
    } finally {
      setLoading(false)
    }
  }

  const renderResults = () => {
    if (loading)                              return <LoadingState mode={activeMode} />
    if (apiError)                             return <ErrorState message={apiError} />
    if (!result)                              return <EmptyState activeMode={activeMode} />
    if (result.status === 'model_not_ready')  return <ModelNotReadyState message={result.message} />
    if (result.status === 'prediction_ready') return <PredictionReadyState result={result} mode={activeMode} />
    return <ErrorState message={result.message || 'Unexpected response received from backend.'} />
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-fade-in">
      {/* Page Header */}
      <header className="space-y-1">
        <h2 className="text-2xl font-bold text-ink-100">Unified Adulteration Detection</h2>
        <p className="text-sm text-ink-400">
          Select a classification pipeline mode and upload your sample file to run prediction.
        </p>
      </header>

      {/* Mode Selector Tabs */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-3" aria-label="Classification mode selector">
        {MODES.map((mode) => {
          const isActive = mode.id === activeMode
          return (
            <button
              key={mode.id}
              onClick={() => handleModeChange(mode.id)}
              disabled={loading}
              className={`
                text-left p-4 rounded-xl border transition-all duration-200 flex flex-col justify-between gap-3
                ${
                  isActive
                    ? 'bg-olive-950/40 border-olive-500/60 shadow-olive-sm ring-1 ring-olive-500/30'
                    : 'bg-surface-900/60 border-surface-700/60 hover:border-surface-600 hover:bg-surface-800/40'
                }
                ${loading ? 'opacity-60 cursor-not-allowed' : ''}
              `}
              aria-pressed={isActive}
            >
              <div className="flex items-center justify-between">
                <span
                  className={`
                    w-8 h-8 rounded-lg flex items-center justify-center transition-colors
                    ${isActive ? 'bg-olive-500/20 text-olive-400' : 'bg-surface-800 text-ink-400'}
                  `}
                >
                  {mode.icon}
                </span>
                {isActive && (
                  <span className="text-[10px] font-semibold text-olive-400 uppercase tracking-wider px-2 py-0.5 rounded-full bg-olive-950 border border-olive-700/50">
                    Active
                  </span>
                )}
              </div>
              <div>
                <h3 className={`text-xs font-bold ${isActive ? 'text-ink-50' : 'text-ink-200'}`}>
                  {mode.title}
                </h3>
                <p className="text-[11px] text-ink-400 mt-0.5 leading-snug">{mode.subtitle}</p>
              </div>
            </button>
          )
        })}
      </section>

      {/* Main Grid: Upload Control & Results Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Left — Upload Panel */}
        <section className="glass-card rounded-xl p-6 space-y-5" aria-label="Sample upload panel">
          <div>
            <h3 className="text-sm font-semibold text-ink-100">{activeMeta?.title}</h3>
            <p className="text-xs text-ink-400 mt-0.5">{activeMeta?.description}</p>
            <p className="text-[11px] text-olive-400 font-mono mt-1.5">{activeMeta?.acceptDesc}</p>
          </div>

          <SampleUploader mode={activeMode} onFileSelected={handleFileSelected} disabled={loading} />

          <button
            id="run-detection-btn"
            onClick={handleSubmit}
            disabled={!selectedFile || loading}
            className="btn-primary w-full justify-center py-2.5 text-xs font-semibold"
            aria-label="Run adulteration detection on uploaded sample"
          >
            {loading ? (
              <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin-slow" />
                Analysing Sample…
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-4 h-4">
                  <path d="M21 21l-4.35-4.35M11 19a8 8 0 1 1 0-16 8 8 0 0 1 0 16z" strokeLinecap="round" />
                </svg>
                Run {activeMeta?.title.split(' ')[0]} Detection
              </>
            )}
          </button>
        </section>

        {/* Right — Results Panel */}
        <section
          className="glass-card rounded-xl p-6 flex flex-col justify-between"
          aria-label="Detection results"
          aria-live="polite"
        >
          <div>
            <h3 className="text-sm font-semibold text-ink-100 mb-4">Prediction Results</h3>
            {renderResults()}
          </div>
        </section>
      </div>
    </div>
  )
}
