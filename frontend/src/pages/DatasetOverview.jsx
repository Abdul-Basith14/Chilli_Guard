import { useEffect, useState } from 'react'
import {
  fetchDatasetStatus,
  triggerDatasetInspection,
  fetchDatasetReport,
  getDatasetChartUrl,
} from '../services/api'
import StatusBadge from '../components/StatusBadge'

export default function DatasetOverview() {
  const [status, setStatus] = useState(null)
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [inspecting, setInspecting] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview') // 'overview' | 'charts' | 'report'

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const statusRes = await fetchDatasetStatus()
      setStatus(statusRes)
      if (statusRes.has_inspection_report) {
        const reportRes = await fetchDatasetReport()
        setReport(reportRes)
      }
    } catch (err) {
      console.error('Failed to load dataset status/report:', err)
      setError(err.response?.data?.detail || 'Failed to connect to backend service.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const handleRunInspection = async () => {
    setInspecting(true)
    setError(null)
    try {
      await triggerDatasetInspection()
      await loadData()
    } catch (err) {
      console.error('Dataset inspection failed:', err)
      setError(err.response?.data?.detail || 'Dataset inspection failed to run.')
    } finally {
      setInspecting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <div className="w-10 h-10 border-4 border-olive-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-ink-300">Loading dataset overview...</p>
      </div>
    )
  }

  const summary = report?.summary || {}
  const dimStats = summary?.dimension_stats || {}
  const classCounts = summary?.class_counts || status?.class_counts || {}

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-olive-subtle pb-5">
        <div>
          <h1 className="text-2xl font-bold text-ink-50 tracking-tight flex items-center gap-3">
            <span>Dataset Overview & Quality Inspection</span>
            {status?.exists ? (
              <StatusBadge status="ready" label={status.structure_type} />
            ) : (
              <StatusBadge status="error" label="Path Missing" />
            )}
          </h1>
          <p className="text-xs text-ink-400 mt-1 font-mono">
            Configured Path: <span className="text-olive-400">{status?.dataset_path}</span>
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRunInspection}
            disabled={inspecting || !status?.exists}
            className="btn-primary flex items-center gap-2 disabled:opacity-50 text-xs px-4 py-2.5"
          >
            {inspecting ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-surface-950 border-t-transparent rounded-full animate-spin" />
                <span>Scanning & Analyzing...</span>
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-4 h-4">
                  <path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  <path d="M9 12l2 2 4-4" />
                </svg>
                <span>{status?.has_inspection_report ? 'Re-Run Inspection' : 'Run Dataset Inspection'}</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error alert */}
      {error && (
        <div className="p-4 bg-red-950/40 border border-red-500/40 rounded-xl text-red-200 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-base">⚠️</span>
            <span>{error}</span>
          </div>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-100">✕</button>
        </div>
      )}

      {/* Missing dataset path empty state */}
      {!status?.exists && (
        <div className="glass-card p-8 text-center space-y-4">
          <div className="text-4xl">📁</div>
          <h3 className="text-base font-semibold text-ink-100">Dataset Directory Not Found</h3>
          <p className="text-xs text-ink-400 max-w-lg mx-auto">
            The configured dataset path <code className="text-olive-400 font-mono">{status?.dataset_path}</code> does not exist on the host server.
            Please set <code className="text-ink-200">DATASET_I_PATH</code> in your <code className="text-ink-200">.env</code> file.
          </p>
        </div>
      )}

      {/* Not inspected state */}
      {status?.exists && !status?.has_inspection_report && !inspecting && (
        <div className="glass-card p-8 text-center space-y-4">
          <div className="text-4xl">🔍</div>
          <h3 className="text-base font-semibold text-ink-100">Dataset Discovered But Not Inspected</h3>
          <p className="text-xs text-ink-400 max-w-lg mx-auto">
            Click <strong>"Run Dataset Inspection"</strong> above to scan file formats, verify image headers, check for duplicates, and generate exploratory reports.
          </p>
        </div>
      )}

      {/* Inspected results dashboard */}
      {status?.has_inspection_report && (
        <>
          {/* Key Metric Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="glass-card p-4 space-y-1">
              <p className="text-[11px] text-ink-400 uppercase font-medium">Discovered</p>
              <p className="text-2xl font-bold text-ink-50">{summary.total_images_discovered?.toLocaleString() || 0}</p>
              <p className="text-[10px] text-ink-500">Total Image Files</p>
            </div>

            <div className="glass-card p-4 space-y-1 border-l-2 border-l-emerald-500">
              <p className="text-[11px] text-ink-400 uppercase font-medium">Readable</p>
              <p className="text-2xl font-bold text-emerald-400">{summary.readable_images?.toLocaleString() || 0}</p>
              <p className="text-[10px] text-ink-500">Passed PIL Header Check</p>
            </div>

            <div className="glass-card p-4 space-y-1 border-l-2 border-l-red-500">
              <p className="text-[11px] text-ink-400 uppercase font-medium">Corrupted</p>
              <p className="text-2xl font-bold text-red-400">{summary.corrupted_images || 0}</p>
              <p className="text-[10px] text-ink-500">Unreadable Files</p>
            </div>

            <div className="glass-card p-4 space-y-1 border-l-2 border-l-amber-500">
              <p className="text-[11px] text-ink-400 uppercase font-medium">Exact Duplicates</p>
              <p className="text-2xl font-bold text-amber-400">{summary.exact_duplicate_groups || 0}</p>
              <p className="text-[10px] text-ink-500">MD5 Hash Groups</p>
            </div>

            <div className="glass-card p-4 space-y-1 border-l-2 border-l-purple-500">
              <p className="text-[11px] text-ink-400 uppercase font-medium">Perceptual Dups</p>
              <p className="text-2xl font-bold text-purple-400">{summary.perceptual_duplicate_groups || 0}</p>
              <p className="text-[10px] text-ink-500">dHash Sets</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex border-b border-olive-subtle gap-4 text-xs font-medium">
            <button
              onClick={() => setActiveTab('overview')}
              className={`pb-2 transition-colors ${
                activeTab === 'overview'
                  ? 'text-olive-400 border-b-2 border-olive-400 font-semibold'
                  : 'text-ink-400 hover:text-ink-200'
              }`}
            >
              Class Breakdown & Quality
            </button>
            <button
              onClick={() => setActiveTab('charts')}
              className={`pb-2 transition-colors ${
                activeTab === 'charts'
                  ? 'text-olive-400 border-b-2 border-olive-400 font-semibold'
                  : 'text-ink-400 hover:text-ink-200'
              }`}
            >
              Generated Visualizations
            </button>
            <button
              onClick={() => setActiveTab('report')}
              className={`pb-2 transition-colors ${
                activeTab === 'report'
                  ? 'text-olive-400 border-b-2 border-olive-400 font-semibold'
                  : 'text-ink-400 hover:text-ink-200'
              }`}
            >
              Executive Markdown Report
            </button>
          </div>

          {/* Tab Content 1: Overview */}
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Class Breakdown */}
              <div className="glass-card p-5 space-y-4">
                <h3 className="text-sm font-semibold text-ink-100 flex items-center justify-between">
                  <span>Class Distribution & Mapping</span>
                  <span className="text-[11px] text-ink-400 font-normal">Dataset I</span>
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left border-collapse">
                    <thead>
                      <tr className="border-b border-olive-subtle text-ink-400">
                        <th className="py-2 font-medium">Class Folder</th>
                        <th className="py-2 font-medium">Mapped Label</th>
                        <th className="py-2 font-medium text-right">Count</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-olive-subtle/50 text-ink-200">
                      {Object.entries(classCounts).map(([folder, count]) => (
                        <tr key={folder} className="hover:bg-surface-850">
                          <td className="py-2.5 font-mono text-olive-300">{folder}</td>
                          <td className="py-2.5 font-medium">
                            {folder === 'C1_PKT' ? 'Pure (Khammam Teja)' : 'Adulterated'}
                          </td>
                          <td className="py-2.5 text-right font-semibold">{count.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Image Specifications & Dimensions */}
              <div className="glass-card p-5 space-y-4">
                <h3 className="text-sm font-semibold text-ink-100">Image Format & Dimensions Summary</h3>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-3 bg-surface-900 rounded-lg space-y-1">
                    <p className="text-ink-400 text-[10px]">Formats Detected</p>
                    <p className="font-semibold text-ink-100">{summary.formats?.join(', ') || 'N/A'}</p>
                  </div>
                  <div className="p-3 bg-surface-900 rounded-lg space-y-1">
                    <p className="text-ink-400 text-[10px]">Color Modes</p>
                    <p className="font-semibold text-ink-100">{summary.color_modes?.join(', ') || 'N/A'}</p>
                  </div>
                  <div className="p-3 bg-surface-900 rounded-lg space-y-1">
                    <p className="text-ink-400 text-[10px]">Width Range</p>
                    <p className="font-semibold text-ink-100">
                      {dimStats.min_width}px – {dimStats.max_width}px (avg {Math.round(dimStats.mean_width || 0)}px)
                    </p>
                  </div>
                  <div className="p-3 bg-surface-900 rounded-lg space-y-1">
                    <p className="text-ink-400 text-[10px]">Height Range</p>
                    <p className="font-semibold text-ink-100">
                      {dimStats.min_height}px – {dimStats.max_height}px (avg {Math.round(dimStats.mean_height || 0)}px)
                    </p>
                  </div>
                </div>

                <div className="p-3 bg-olive-950/40 border border-olive-500/20 rounded-lg text-[11px] text-ink-300">
                  💡 <strong>Preprocessing note:</strong> All images will be resized to 224×224 pixels during training.
                </div>
              </div>
            </div>
          )}

          {/* Tab Content 2: Visualizations */}
          {activeTab === 'charts' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="glass-card p-4 space-y-3">
                <h3 className="text-sm font-semibold text-ink-100">Class Distribution Chart</h3>
                <div className="bg-surface-950 p-2 rounded-lg border border-olive-subtle overflow-hidden">
                  <img
                    src={getDatasetChartUrl('class_distribution.png')}
                    alt="Class Distribution Chart"
                    className="w-full h-auto rounded"
                    onError={(e) => {
                      e.target.style.display = 'none'
                    }}
                  />
                </div>
              </div>

              <div className="glass-card p-4 space-y-3">
                <h3 className="text-sm font-semibold text-ink-100">Sample Image Preview Grid</h3>
                <div className="bg-surface-950 p-2 rounded-lg border border-olive-subtle overflow-hidden">
                  <img
                    src={getDatasetChartUrl('sample_images.png')}
                    alt="Sample Image Grid"
                    className="w-full h-auto rounded"
                    onError={(e) => {
                      e.target.style.display = 'none'
                    }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Tab Content 3: Executive Markdown Report */}
          {activeTab === 'report' && (
            <div className="glass-card p-6 space-y-4">
              <h3 className="text-sm font-semibold text-ink-100 border-b border-olive-subtle pb-3">
                Executive Inspection Report (<code className="text-olive-400">dataset_inspection_report.md</code>)
              </h3>
              <pre className="text-xs text-ink-200 font-mono bg-surface-950 p-4 rounded-xl border border-olive-subtle whitespace-pre-wrap leading-relaxed overflow-x-auto">
                {report?.markdown_report}
              </pre>
            </div>
          )}
        </>
      )}
    </div>
  )
}
