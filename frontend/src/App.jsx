import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar.jsx'
import Dashboard from './pages/Dashboard.jsx'
import DatasetOverview from './pages/DatasetOverview.jsx'
import ImageDetection from './pages/ImageDetection.jsx'
import ModelPerformance from './pages/ModelPerformance.jsx'
import About from './pages/About.jsx'

export default function App() {
  return (
    <div className="flex min-h-screen bg-surface-950">
      {/* Persistent sidebar navigation */}
      <Sidebar />

      {/* Main content area */}
      <main className="flex-1 flex flex-col min-h-screen overflow-x-hidden">
        {/* Subtle top ambient glow */}
        <div
          className="pointer-events-none fixed top-0 left-0 right-0 h-64 z-0"
          style={{
            background:
              'radial-gradient(ellipse at 50% -10%, rgba(125,143,40,0.12) 0%, transparent 70%)',
          }}
        />

        <div className="relative z-10 flex-1 p-6 lg:p-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dataset" element={<DatasetOverview />} />
            <Route path="/detect" element={<ImageDetection />} />
            <Route path="/performance" element={<ModelPerformance />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}
