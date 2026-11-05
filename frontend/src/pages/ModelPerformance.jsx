import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import StatusBadge from '../components/StatusBadge.jsx'
import { fetchModelStatus, fetchModelMetrics } from '../services/api.js'

function MetricRow({ label, value, highlight = false }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-surface-700/60 last:border-0">
      <span className="text-xs text-ink-400">{label}</span>
      <span className={`text-xs font-mono font-semibold ${highlight ? 'text-primary-400' : 'text-ink-200'}`}>
        {value !== null && value !== undefined
          ? typeof value === 'number'
            ? value.toFixed(4)
            : value
          : 'N/A'}
      </span>
    </div>
  )
}

export default function ModelPerformance() {
  const [modelStatus, setModelStatus] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadData() {
      try {
        const [statusRes, metricsRes] = await Promise.all([
          fetchModelStatus().catch(() => null),
          fetchModelMetrics().catch(() => null),
        ])
        setModelStatus(statusRes)
        setMetrics(metricsRes)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  const hasMetrics = metrics?.status === 'ready'
  const expA = metrics?.exp_a
  const expB = metrics?.exp_b
  const testRes = metrics?.test_results
  const audit = metrics?.audit

  return (
    <div className="max-w-4xl mx-auto animate-fade-in space-y-8">
      {/* Header */}
      <header>
        <h2 className="text-2xl font-bold text-ink-100">Model Performance</h2>
        <p className="text-sm text-ink-500 mt-1">
          Phase 2 Binary Image Classifier (ResNet18 Pre-trained Backbone)
        </p>
      </header>

      {/* Model status summary */}
      <div className="glass-card rounded-xl p-6 flex flex-col sm:flex-row items-start sm:items-center gap-4">
        <div className="w-12 h-12 rounded-full bg-surface-700/60 flex items-center justify-center shrink-0">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="w-6 h-6 text-primary-400">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <p className="text-sm font-semibold text-ink-200">
              {hasMetrics ? 'Model Trained & Ready' : 'Model Status'}
            </p>
            <StatusBadge status={modelStatus?.status || (hasMetrics ? 'ready' : 'not_trained')} size="sm" />
          </div>
          <p className="text-xs text-ink-500 leading-relaxed">
            {hasMetrics
              ? `ResNet18 binary classifier trained on Dataset I. Controlled experiments evaluated unweighted loss (Exp A) vs class-weighted loss (Exp B).`
              : 'No trained model checkpoint found yet. Run Phase 2 training to populate metrics.'}
          </p>
        </div>
      </div>

      {/* Leakage audit callout */}
      {audit && (
        <div className="glass-card rounded-xl p-5 border border-primary-500/20 bg-primary-500/5">
          <h3 className="text-xs font-semibold text-primary-400 uppercase tracking-wider mb-2">
            Data Leakage Audit & Deduplication
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            <div className="p-2 rounded bg-surface-800/40">
              <p className="text-lg font-bold text-ink-100">{audit.exact_cross_split_groups}</p>
              <p className="text-[10px] text-ink-500">Cross-Split Dupes</p>
            </div>
            <div className="p-2 rounded bg-surface-800/40">
              <p className="text-lg font-bold font-mono text-amber-400">{audit.train_side_excluded_count}</p>
              <p className="text-[10px] text-ink-500">Excluded (Train Copy)</p>
            </div>
            <div className="p-2 rounded bg-surface-800/40">
              <p className="text-lg font-bold text-emerald-400">278</p>
              <p className="text-[10px] text-ink-500">Test Set (Untouched)</p>
            </div>
            <div className="p-2 rounded bg-surface-800/40">
              <p className="text-lg font-bold text-primary-400">100%</p>
              <p className="text-[10px] text-ink-500">Split Isolation</p>
            </div>
          </div>
        </div>
      )}

      {/* Experiments side-by-side comparison */}
      {hasMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Exp A */}
          <section className="glass-card rounded-xl p-5" aria-label="Exp A metrics">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-semibold text-ink-300 uppercase tracking-wider">
                Exp A: Standard Loss
              </h3>
              <span className="text-[10px] text-ink-500 font-mono">Unweighted</span>
            </div>
            {expA ? (
              <>
                <MetricRow label="Validation Accuracy" value={expA.val_accuracy} />
                <MetricRow label="Balanced Accuracy" value={expA.val_balanced_accuracy} />
                <MetricRow label="Macro F1-Score" value={expA.val_macro_f1} highlight />
                <MetricRow label="AUC-ROC" value={expA.val_auc_roc} />
                <MetricRow label="Best Epoch" value={expA.best_epoch} />
              </>
            ) : (
              <p className="text-xs text-ink-500">Exp A results unavailable.</p>
            )}
          </section>

          {/* Exp B */}
          <section className="glass-card rounded-xl p-5 border border-primary-500/30" aria-label="Exp B metrics">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-semibold text-primary-400 uppercase tracking-wider">
                Exp B: Weighted Loss (Recommended)
              </h3>
              <span className="text-[10px] text-primary-400 font-mono font-semibold">Weighted</span>
            </div>
            {expB ? (
              <>
                <MetricRow label="Validation Accuracy" value={expB.val_accuracy} />
                <MetricRow label="Balanced Accuracy" value={expB.val_balanced_accuracy} />
                <MetricRow label="Macro F1-Score" value={expB.val_macro_f1} highlight />
                <MetricRow label="AUC-ROC" value={expB.val_auc_roc} />
                <MetricRow label="Best Epoch" value={expB.best_epoch} />
              </>
            ) : (
              <p className="text-xs text-ink-500">Exp B results unavailable.</p>
            )}
          </section>
        </div>
      )}

      {/* Final test set performance */}
      {testRes && (
        <section className="glass-card rounded-xl p-5" aria-label="Final test evaluation">
          <h3 className="text-xs font-semibold text-ink-300 uppercase tracking-wider mb-4">
            Final Held-Out Test Set Performance (278 Images)
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-3 rounded-lg bg-surface-800/60 border border-surface-700/60">
              <p className="text-[10px] text-ink-500 uppercase">Test Accuracy</p>
              <p className="text-xl font-mono font-bold text-ink-100">{testRes.accuracy ? (testRes.accuracy * 100).toFixed(1) + '%' : 'N/A'}</p>
            </div>
            <div className="p-3 rounded-lg bg-surface-800/60 border border-surface-700/60">
              <p className="text-[10px] text-ink-500 uppercase">Balanced Acc</p>
              <p className="text-xl font-mono font-bold text-ink-100">{testRes.balanced_accuracy ? (testRes.balanced_accuracy * 100).toFixed(1) + '%' : 'N/A'}</p>
            </div>
            <div className="p-3 rounded-lg bg-surface-800/60 border border-surface-700/60">
              <p className="text-[10px] text-ink-500 uppercase">Macro F1</p>
              <p className="text-xl font-mono font-bold text-primary-400">{testRes.f1_macro ? testRes.f1_macro.toFixed(4) : 'N/A'}</p>
            </div>
            <div className="p-3 rounded-lg bg-surface-800/60 border border-surface-700/60">
              <p className="text-[10px] text-ink-500 uppercase">AUC-ROC</p>
              <p className="text-xl font-mono font-bold text-emerald-400">{testRes.auc_roc ? testRes.auc_roc.toFixed(4) : 'N/A'}</p>
            </div>
          </div>
        </section>
      )}

      <div className="text-center">
        <Link to="/detect" id="perf-to-detect" className="btn-primary text-xs">
          Test Live Image Detection
        </Link>
      </div>
    </div>
  )
}
