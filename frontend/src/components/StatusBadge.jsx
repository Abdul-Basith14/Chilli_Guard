/**
 * StatusBadge — small pill that visualises a status with colour coding.
 *
 * Props:
 *   status  : 'ready' | 'not_trained' | 'loading_failed' | 'ok' | 'error' | string
 *   label   : override display text (defaults to capitalised status)
 *   size    : 'sm' | 'md' (default 'md')
 */

const STATUS_STYLES = {
  ok:             'bg-olive-900/50 text-olive-300 border border-olive-700/40',
  ready:          'bg-olive-900/50 text-olive-300 border border-olive-700/40',
  not_trained:    'bg-surface-700 text-ink-400 border border-surface-600',
  loading_failed: 'bg-red-950/60 text-red-400 border border-red-900/40',
  error:          'bg-red-950/60 text-red-400 border border-red-900/40',
  loading:        'bg-amber-950/60 text-amber-400 border border-amber-900/40',
}

const DOT_STYLES = {
  ok:             'bg-olive-400',
  ready:          'bg-olive-400',
  not_trained:    'bg-ink-500',
  loading_failed: 'bg-red-400',
  error:          'bg-red-400',
  loading:        'bg-amber-400 animate-pulse-olive',
}

function humanise(status) {
  return (status || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

export default function StatusBadge({ status = 'not_trained', label, size = 'md' }) {
  const style = STATUS_STYLES[status] ?? STATUS_STYLES.not_trained
  const dot   = DOT_STYLES[status]   ?? DOT_STYLES.not_trained
  const text  = label ?? humanise(status)

  return (
    <span
      className={`status-badge ${style} ${size === 'sm' ? 'text-[10px]' : 'text-xs'}`}
      role="status"
      aria-label={`Status: ${text}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} aria-hidden="true" />
      {text}
    </span>
  )
}
