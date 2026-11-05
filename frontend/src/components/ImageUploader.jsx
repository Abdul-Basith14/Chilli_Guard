/**
 * ImageUploader — drag-and-drop / file-select image upload component.
 *
 * Props:
 *   onFileSelected(file: File) — called when a valid image is chosen
 *   disabled: boolean           — disables interaction while loading
 */

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

const ACCEPTED_TYPES = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png':  ['.png'],
  'image/webp': ['.webp'],
  'image/bmp':  ['.bmp'],
  'image/tiff': ['.tiff', '.tif'],
}

const MAX_SIZE_MB = 10
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

// ── Upload icon ────────────────────────────────────────────────────────────────
function UploadIcon({ className = 'w-10 h-10' }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.4} className={className}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" strokeLinecap="round" strokeLinejoin="round" />
      <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round" />
    </svg>
  )
}

export default function ImageUploader({ onFileSelected, disabled = false }) {
  const [preview, setPreview] = useState(null)
  const [fileName, setFileName] = useState(null)
  const [sizeLabel, setSizeLabel] = useState(null)
  const [localError, setLocalError] = useState(null)

  const handleFile = useCallback(
    (file) => {
      setLocalError(null)
      if (!file) return

      // Size guard (backend also validates, but give fast UX feedback)
      if (file.size > MAX_SIZE_BYTES) {
        setLocalError(`File too large. Maximum allowed size is ${MAX_SIZE_MB} MB.`)
        return
      }

      const url = URL.createObjectURL(file)
      setPreview(url)
      setFileName(file.name)
      setSizeLabel(`${(file.size / 1024).toFixed(1)} KB`)
      onFileSelected(file)
    },
    [onFileSelected]
  )

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE_BYTES,
    multiple: false,
    disabled,
    onDropAccepted: ([file]) => handleFile(file),
    onDropRejected: (fileRejections) => {
      const reason = fileRejections[0]?.errors[0]?.message || 'Invalid file.'
      setLocalError(reason)
    },
  })

  const handleClear = () => {
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setFileName(null)
    setSizeLabel(null)
    setLocalError(null)
  }

  const borderClass = isDragReject
    ? 'border-red-600 bg-red-950/20'
    : isDragActive
    ? 'border-olive-400 bg-olive-950/30 shadow-olive-sm'
    : 'border-surface-600 hover:border-olive-600 hover:bg-olive-subtle'

  return (
    <div className="w-full space-y-4 animate-fade-in">
      {/* Drop zone */}
      {!preview ? (
        <div
          {...getRootProps()}
          id="image-dropzone"
          className={`
            relative flex flex-col items-center justify-center gap-3
            border-2 border-dashed rounded-xl p-10 cursor-pointer
            transition-all duration-200 ease-in-out
            ${borderClass}
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
          aria-label="Image upload drop zone"
        >
          <input {...getInputProps()} id="image-file-input" aria-label="Upload image file" />

          <div className={`text-ink-400 transition-transform duration-200 ${isDragActive ? 'scale-110 text-olive-400' : ''}`}>
            <UploadIcon className="w-10 h-10" />
          </div>

          <div className="text-center">
            <p className="text-sm font-medium text-ink-200">
              {isDragActive ? 'Drop your image here' : 'Drag & drop an image here'}
            </p>
            <p className="text-xs text-ink-500 mt-1">
              or <span className="text-olive-400 underline underline-offset-2">browse files</span>
            </p>
            <p className="text-[10px] text-ink-500/60 mt-2">
              JPEG, PNG, WebP, BMP, TIFF — max {MAX_SIZE_MB} MB
            </p>
          </div>
        </div>
      ) : (
        /* Image preview */
        <div className="glass-card rounded-xl overflow-hidden animate-slide-up">
          <div className="relative aspect-video bg-surface-800 flex items-center justify-center">
            <img
              src={preview}
              alt="Uploaded chilli powder sample preview"
              className="max-h-64 max-w-full object-contain"
            />
            {/* Remove button */}
            <button
              onClick={handleClear}
              disabled={disabled}
              className="absolute top-2 right-2 p-1.5 rounded-lg bg-surface-900/80 text-ink-400
                         hover:text-red-400 hover:bg-red-950/40 transition-colors duration-150
                         disabled:opacity-40"
              aria-label="Remove selected image"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-4 h-4">
                <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
              </svg>
            </button>
          </div>

          {/* File meta */}
          <div className="px-4 py-2.5 flex items-center justify-between border-t border-olive-subtle">
            <div className="flex items-center gap-2 min-w-0">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="w-3.5 h-3.5 text-olive-400 shrink-0">
                <rect x="5" y="2" width="14" height="20" rx="2" />
                <path d="M9 7h6M9 11h6M9 15h4" strokeLinecap="round" />
              </svg>
              <span className="text-xs text-ink-300 truncate">{fileName}</span>
            </div>
            <span className="text-[10px] text-ink-500 shrink-0 ml-2">{sizeLabel}</span>
          </div>
        </div>
      )}

      {/* Validation error */}
      {localError && (
        <p className="text-xs text-red-400 flex items-center gap-1.5 animate-fade-in" role="alert">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-3.5 h-3.5 shrink-0">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 8v4M12 16h.01" strokeLinecap="round" />
          </svg>
          {localError}
        </p>
      )}
    </div>
  )
}
