/**
 * Dashboard — landing page for ChilliGuard.
 *
 * Shows project overview, real-time backend health status,
 * and model readiness. No fake data is displayed.
 */

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchHealth, fetchModelStatus } from '../services/api.js'
import StatusBadge from '../components/StatusBadge.jsx'

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({ title, value, sub, accent = false }) {
  return (
    <div className={`glass-card glass-card-hover rounded-xl p-5 ${accent ? 'border-olive-700/30' : ''}`}>
      <p className="text-xs text-ink-500 uppercase tracking-widest font-medium mb-1">{title}</p>
      <p className={`text-2xl font-bold ${accent ? 'text-gradient-olive' : 'text-ink-100'}`}>{value}</p>
      {sub && <p className="text-xs text-ink-500 mt-1">{sub}</p>}
    </div>
  )
}

// ── Phase card ────────────────────────────────────────────────────────────────
function PhaseCard({ num, title, status, description }) {
  const isActive  = status === 'active'
  const isDone    = status === 'done'
  const isPending = status === 'pending'

  return (
    <div className={`
      flex gap-4 p-4 rounded-xl border transition-colors duration-150
      ${isActive  ? 'bg-olive-subtle border-olive-700/40' : ''}
      ${isDone    ? 'border-surface-700 bg-surface-800/40' : ''}
      ${isPending ? 'border-surface-800 opacity-50' : ''}
    `}>
      <div className={`
        w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0 mt-0.5
        ${isActive  ? 'bg-olive-700 text-olive-100' : ''}
        ${isDone    ? 'bg-surface-600 text-ink-400' : ''}
        ${isPending ? 'bg-surface-800 text-ink-600' : ''}
      `}>
        {isDone ? '✓' : num}
      </div>
      <div>
        <p className={`text-sm font-semibold ${isActive ? 'text-olive-300' : isDone ? 'text-ink-300' : 'text-ink-600'}`}>
          {title}
        </p>
        <p className="text-xs text-ink-500 mt-0.5 leading-relaxed">{description}</p>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [health, setHealth]         = useState(null)
  const [modelStatus, setModel]     = useState(null)
  const [healthLoading, setHL]      = useState(true)
  const [modelLoading, setML]       = useState(true)
  const [healthError, setHE]        = useState(false)
  const [modelError, setME]         = useState(false)

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHE(true))
      .finally(() => setHL(false))

    fetchModelStatus()
      .then(setModel)
      .catch(() => setME(true))
      .finally(() => setML(false))
  }, [])

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-fade-in">

      {/* Page header */}
      <header>
        <h2 className="text-3xl font-extrabold text-gradient-olive tracking-tight">
          ChilliGuard
        </h2>
        <p className="text-sm text-ink-400 mt-1 max-w-xl leading-relaxed">
          Hybrid Machine Learning for Red Chilli Powder Adulteration Detection —
          Image-based classification with future multi-modal data fusion.
        </p>
      </header>

      {/* Status bar */}
      <section
        className="glass-card rounded-xl p-4 flex flex-wrap items-center gap-6"
        aria-label="System status"
      >
        {/* Backend health */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-ink-500">Backend</span>
          {healthLoading ? (
            <StatusBadge status="loading" label="Connecting…" />
          ) : healthError ? (
            <StatusBadge status="error" label="Unreachable" />
          ) : (
            <StatusBadge status={health?.status === 'ok' ? 'ok' : 'error'} />
          )}
        </div>

        <div className="w-px h-4 bg-surface-600" aria-hidden="true" />

        {/* Model status */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-ink-500">Model</span>
          {modelLoading ? (
            <StatusBadge status="loading" label="Checking…" />
          ) : modelError ? (
            <StatusBadge status="error" label="Unavailable" />
          ) : (
            <StatusBadge status={modelStatus?.status} />
          )}
        </div>

        {modelStatus?.message && !modelLoading && !modelError && (
          <>
            <div className="w-px h-4 bg-surface-600" aria-hidden="true" />
            <p className="text-xs text-ink-500 flex-1">{modelStatus.message}</p>
          </>
        )}
      </section>

      {/* Stats row */}
      <section
        className="grid grid-cols-2 md:grid-cols-4 gap-4"
        aria-label="Dataset statistics"
      >
        <StatCard title="Dataset I" value="1,339" sub="Binary classification" accent />
        <StatCard title="Classes (I)" value="2" sub="Pure · Adulterated" />
        <StatCard title="Dataset II" value="5,275" sub="19-class (Phase 3)" />
        <StatCard title="Current Phase" value="0" sub="Initial scaffold" accent />
      </section>

      {/* CTA */}
      <section className="flex flex-wrap gap-3">
        <Link
          to="/detect"
          id="cta-detect"
          className="btn-primary"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-4 h-4">
            <path d="M21 21l-4.35-4.35M11 19a8 8 0 1 1 0-16 8 8 0 0 1 0 16z" strokeLinecap="round" />
          </svg>
          Try Image Detection
        </Link>
        <Link to="/about" id="cta-about" className="btn-secondary">
          Learn More
        </Link>
      </section>

      {/* Research roadmap */}
      <section aria-labelledby="roadmap-heading">
        <h3 id="roadmap-heading" className="text-sm font-semibold text-ink-300 uppercase tracking-widest mb-4">
          Research Roadmap
        </h3>
        <div className="space-y-2">
          <PhaseCard num="0" status="active"  title="Project Scaffold"        description="Frontend, backend, ML structure. No model trained yet." />
          <PhaseCard num="1" status="pending" title="Dataset Inspection & EDA" description="Image counts, class distributions, sample visualisations." />
          <PhaseCard num="2" status="pending" title="Binary Classifier"        description="CNN/ViT on Dataset I — Pure vs Adulterated." />
          <PhaseCard num="3" status="pending" title="19-Class Classifier"      description="Multi-class model on Dataset II." />
          <PhaseCard num="4" status="pending" title="Multi-Modal Fusion"       description="Images + FT-IR / Raman / sensor data (when available)." />
          <PhaseCard num="5" status="pending" title="Explainable AI"           description="Grad-CAM, SHAP, LIME visualisations." />
        </div>
      </section>
    </div>
  )
}
