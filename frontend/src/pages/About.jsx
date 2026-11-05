/**
 * About — project context, dataset information, and methodology overview.
 */

export default function About() {
  return (
    <div className="max-w-3xl mx-auto animate-fade-in space-y-8">

      {/* Header */}
      <header>
        <h2 className="text-2xl font-bold text-ink-100">About</h2>
        <p className="text-sm text-ink-500 mt-1">
          Project context, dataset information, and research methodology.
        </p>
      </header>

      {/* Project overview */}
      <section className="glass-card rounded-xl p-6 space-y-3" aria-labelledby="about-project">
        <h3 id="about-project" className="text-sm font-semibold text-olive-300">
          Project Overview
        </h3>
        <p className="text-sm text-ink-400 leading-relaxed">
          <strong className="text-ink-200">ChilliGuard</strong> is an academic research project that applies
          hybrid machine learning to detect adulteration in red chilli powder. The initial phase focuses on
          binary image classification (Pure vs Adulterated) using convolutional neural networks and vision
          transformers. Future phases will integrate FT-IR / Raman spectroscopy and sensor data for
          multi-modal fusion.
        </p>
      </section>

      {/* Dataset */}
      <section className="glass-card rounded-xl p-6" aria-labelledby="about-dataset">
        <h3 id="about-dataset" className="text-sm font-semibold text-olive-300 mb-4">Dataset</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-surface-700">
                <th className="text-left py-2 text-ink-400 font-medium w-1/4">Dataset</th>
                <th className="text-left py-2 text-ink-400 font-medium">Images</th>
                <th className="text-left py-2 text-ink-400 font-medium">Classes</th>
                <th className="text-left py-2 text-ink-400 font-medium">Task</th>
                <th className="text-left py-2 text-ink-400 font-medium">Phase</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-surface-800">
                <td className="py-2.5 text-ink-200 font-medium">Dataset I</td>
                <td className="py-2.5 text-ink-300">1,339</td>
                <td className="py-2.5 text-ink-300">2 (C1_PKT, C2_AKT)</td>
                <td className="py-2.5 text-ink-300">Binary classification</td>
                <td className="py-2.5 text-olive-400">Phase 2</td>
              </tr>
              <tr>
                <td className="py-2.5 text-ink-200 font-medium">Dataset II</td>
                <td className="py-2.5 text-ink-300">5,275</td>
                <td className="py-2.5 text-ink-300">19</td>
                <td className="py-2.5 text-ink-300">Multi-class classification</td>
                <td className="py-2.5 text-ink-500">Phase 3</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="mt-4 bg-olive-subtle border border-olive-700/30 rounded-lg px-4 py-3">
          <p className="text-[11px] text-olive-400/80">
            <strong>C1_PKT</strong> — Pure chilli powder samples ·{' '}
            <strong>C2_AKT</strong> — Adulterated samples
          </p>
        </div>
      </section>

      {/* Methodology */}
      <section className="glass-card rounded-xl p-6 space-y-3" aria-labelledby="about-method">
        <h3 id="about-method" className="text-sm font-semibold text-olive-300">Methodology</h3>
        <ul className="space-y-2 text-sm text-ink-400">
          {[
            ['Image Classification', 'CNN (ResNet50 / EfficientNet) and Vision Transformer (ViT) with transfer learning'],
            ['Data Augmentation', 'Flips, rotation, colour jitter, Gaussian blur — designed for powder imagery'],
            ['Evaluation', 'Accuracy, Precision, Recall, F1 (macro & weighted), AUC-ROC, Confusion Matrix'],
            ['Explainability', 'Grad-CAM saliency maps, SHAP feature importance (Phase 5)'],
            ['Multi-Modal Fusion', 'FT-IR / Raman spectroscopy + sensor data integration (when available — Phase 4)'],
          ].map(([title, desc]) => (
            <li key={title} className="flex gap-2">
              <span className="text-olive-600 mt-0.5 shrink-0">▸</span>
              <span>
                <strong className="text-ink-300">{title}:</strong> {desc}
              </span>
            </li>
          ))}
        </ul>
      </section>

      {/* Tech stack */}
      <section className="glass-card rounded-xl p-6" aria-labelledby="about-stack">
        <h3 id="about-stack" className="text-sm font-semibold text-olive-300 mb-4">Technology Stack</h3>
        <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-xs text-ink-400">
          {[
            ['Frontend', 'React 18 + Vite + Tailwind CSS'],
            ['Backend', 'Python · FastAPI · Pydantic'],
            ['ML Framework', 'PyTorch · torchvision'],
            ['Classical ML', 'scikit-learn'],
            ['Image Processing', 'Pillow'],
            ['Visualisation', 'matplotlib · seaborn'],
          ].map(([layer, tech]) => (
            <div key={layer} className="flex gap-2">
              <span className="text-ink-600 shrink-0">{layer}:</span>
              <span className="text-ink-300">{tech}</span>
            </div>
          ))}
        </div>
      </section>

      <p className="text-xs text-ink-600 text-center pb-4">
        Phase 0 — Initial Scaffold · No model trained · No spectral data available
      </p>
    </div>
  )
}
