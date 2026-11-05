import { NavLink } from 'react-router-dom'

// ── Icon SVGs (inline, no dependency) ────────────────────────────────────────
const icons = {
  dashboard: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="w-4 h-4 shrink-0">
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </svg>
  ),
  dataset: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="w-4 h-4 shrink-0">
      <path d="M4 7v10c0 2 1.5 3 3.5 3h9c2 0 3.5-1 3.5-3V7" strokeLinecap="round" />
      <ellipse cx="12" cy="7" rx="8" ry="3" />
      <path d="M4 12c0 1.65 3.58 3 8 3s8-1.35 8-3" />
    </svg>
  ),
  detect: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="w-4 h-4 shrink-0">
      <path d="M21 21l-4.35-4.35M11 19a8 8 0 1 1 0-16 8 8 0 0 1 0 16z" strokeLinecap="round" />
    </svg>
  ),
  performance: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="w-4 h-4 shrink-0">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  about: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="w-4 h-4 shrink-0">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" strokeLinecap="round" />
      <line x1="12" y1="16" x2="12.01" y2="16" strokeLinecap="round" />
    </svg>
  ),
}

const navItems = [
  { to: '/', label: 'Dashboard', icon: icons.dashboard, end: true },
  { to: '/dataset', label: 'Dataset Overview', icon: icons.dataset },
  { to: '/detect', label: 'Sample Detection', icon: icons.detect },
  { to: '/performance', label: 'Model Performance', icon: icons.performance },
  { to: '/about', label: 'About', icon: icons.about },
]

export default function Sidebar() {
  return (
    <aside className="w-60 shrink-0 flex flex-col bg-surface-900 border-r border-olive-subtle min-h-screen">

      {/* Logo / brand */}
      <div className="px-5 py-6 border-b border-olive-subtle">
        <div className="flex items-center gap-2.5">
          {/* Chilli emoji as a lightweight logo */}
          <span className="text-2xl select-none" aria-hidden="true">🌶️</span>
          <div>
            <h1 className="text-sm font-bold text-ink-50 tracking-tight leading-none">
              ChilliGuard
            </h1>
            <p className="text-[10px] text-ink-500 mt-0.5 leading-none">
              Adulteration Detection
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1" aria-label="Main navigation">
        {navItems.map(({ to, label, icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              isActive ? 'nav-link-active' : 'nav-link'
            }
          >
            {icon}
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-olive-subtle">
        <p className="text-[10px] text-ink-500 leading-relaxed">
          Phase 5 — Unified Detection System
        </p>
        <p className="text-[10px] text-ink-500/60 mt-0.5">
          Multi-Mode Inference
        </p>
      </div>
    </aside>
  )
}
