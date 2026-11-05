/**
 * SampleUploader — drag-and-drop / file-select component supporting
 * image uploads (Binary / 19-Class mode) and spectral file uploads (FTIR mode).
 *
 * Props:
 *   mode: 'binary' | 'multiclass' | 'ftir'
 *   onFileSelected(file: File) — called when a valid sample is chosen
 *   disabled: boolean           — disables interaction while loading
 */

import { useCallback, useEffect, useState } from 'react'
import { useDropzone } from 'react-dropzone'

const IMAGE_TYPES = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/webp': ['.webp'],
  'image/bmp': ['.bmp'],
  'image/tiff': ['.tiff', '.tif'],
}

const SPECTRAL_TYPES = {
  'text/csv': ['.csv'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
  'application/vnd.ms-excel': ['.xls'],
  'application/x-python-ndarray': ['.npy'],
  'application/octet-stream': ['.npy', '.csv', '.xlsx', '.xls'],
}

const MAX_SIZE_MB = 10
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

function UploadIcon({ className = 'w-10 h-10' }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.4} className={className}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" strokeLinecap="round" strokeLinejoin="round" />
      <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round" />
    </svg>
  )
}

function SpectralFileIcon({ className = 'w-10 h-10' }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.4} className={className}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeLinecap="round" strokeLinejoin="round" />
      <polyline points="14 2 14 8 20 8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M8 13l2-2 3 4 3-3 2 2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export default function SampleUploader({ mode = 'binary', onFileSelected, disabled = false }) {
  const [previewUrl, setPreviewUrl] = useState(null)
  const [fileMeta, setFileMeta] = useState(null)
  const [localError, setLocalError] = useState(null)

  const isSpectralMode = mode === 'ftir'
  const acceptedTypes = isSpectralMode ? SPECTRAL_TYPES : IMAGE_TYPES

  // Clear selection when mode switches
  useEffect(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setFileMeta(null)
    setLocalError(null)
  }, [mode])

  const handleFile = useCallback(
    (file) => {
      setLocalError(null)
      if (!file) return

      if (file.size > MAX_SIZE_BYTES) {
        setLocalError(`File size exceeds maximum limit of ${MAX_SIZE_MB} MB.`)
        return
      }

      const ext = file.name.split('.').pop()?.toLowerCase()
      if (isSpectralMode) {
        if (!['csv', 'xlsx', 'xls', 'npy'].includes(ext)) {
          setLocalError(`Unsupported spectral extension '.${ext}'. Expected .csv, .xlsx, .xls, or .npy.`)
          return
        }
        setPreviewUrl(null)
      } else {
        if (!['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff'].includes(ext)) {
          setLocalError(`Unsupported image extension '.${ext}'. Expected JPG, PNG, WebP, BMP, or TIFF.`)
          return
        }
        const url = URL.createObjectURL(file)
        setPreviewUrl(url)
      }

      setFileMeta({
        name: file.name,
        sizeLabel: `${(file.size / 1024).toFixed(1)} KB`,
        ext: ext?.toUpperCase(),
      })
      onFileSelected(file)
    },
    [isSpectralMode, onFileSelected]
  )

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    accept: acceptedTypes,
    maxSize: MAX_SIZE_BYTES,
    multiple: false,
    disabled,
    onDropAccepted: ([file]) => handleFile(file),
    onDropRejected: (fileRejections) => {
      const reason = fileRejections[0]?.errors[0]?.message || 'Invalid file format.'
      setLocalError(reason)
    },
  })

  const handleClear = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setFileMeta(null)
    setLocalError(null)
  }

  const borderClass = isDragReject
    ? 'border-red-600 bg-red-950/20'
    : isDragActive
    ? 'border-olive-400 bg-olive-950/30 shadow-olive-sm'
    : 'border-surface-600 hover:border-olive-600 hover:bg-olive-subtle'

  return (
    <div className="w-full space-y-4 animate-fade-in">
      {!fileMeta ? (
        /* Dropzone */
        <div
          {...getRootProps()}
          id="sample-dropzone"
          className={`
            relative flex flex-col items-center justify-center gap-3
            border-2 border-dashed rounded-xl p-8 lg:p-10 cursor-pointer
            transition-all duration-200 ease-in-out
            ${borderClass}
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
          aria-label="Sample upload drop zone"
        >
          <input {...getInputProps()} id="sample-file-input" aria-label="Upload sample file" />

          <div className={`text-ink-400 transition-transform duration-200 ${isDragActive ? 'scale-110 text-olive-400' : ''}`}>
            {isSpectralMode ? <SpectralFileIcon className="w-10 h-10" /> : <UploadIcon className="w-10 h-10" />}
          </div>

          <div className="text-center space-y-1">
            <p className="text-sm font-medium text-ink-200">
              {isDragActive
                ? 'Drop file here'
                : isSpectralMode
                ? 'Drag & drop spectral data file'
                : 'Drag & drop chilli image file'}
            </p>
            <p className="text-xs text-ink-500">
              or <span className="text-olive-400 underline underline-offset-2">browse files</span>
            </p>
            <p className="text-[10px] text-ink-500/70 mt-1">
              {isSpectralMode
                ? 'ATR-FTIR files: .CSV, .XLSX, .XLS, .NPY — Max 10 MB'
                : 'JPEG, PNG, WebP, BMP, TIFF — Max 10 MB'}
            </p>
          </div>
        </div>
      ) : (
        /* Preview Card */
        <div className="glass-card rounded-xl overflow-hidden animate-slide-up">
          {!isSpectralMode && previewUrl ? (
            /* Image Preview */
            <div className="relative aspect-video bg-surface-800 flex items-center justify-center">
              <img
                src={previewUrl}
                alt="Uploaded chilli powder sample preview"
                className="max-h-60 max-w-full object-contain"
              />
              <button
                onClick={handleClear}
                disabled={disabled}
                className="absolute top-2 right-2 p-1.5 rounded-lg bg-surface-900/80 text-ink-400
                           hover:text-red-400 hover:bg-red-950/40 transition-colors duration-150
                           disabled:opacity-40"
                aria-label="Remove selected file"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-4 h-4">
                  <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
                </svg>
              </button>
            </div>
          ) : (
            /* Spectral File Summary Card */
            <div className="p-6 bg-surface-800/80 flex items-center justify-between">
              <div className="flex items-center gap-3.5 min-w-0">
                <div className="w-12 h-12 rounded-lg bg-olive-950/60 border border-olive-700/40 flex items-center justify-center shrink-0">
                  <SpectralFileIcon className="w-6 h-6 text-olive-400" />
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-semibold text-ink-200 truncate">{fileMeta.name}</p>
                  <p className="text-[10px] text-ink-400 mt-0.5">
                    ATR-FTIR Spectral Input · <span className="text-olive-400 font-mono">{fileMeta.ext}</span> format
                  </p>
                </div>
              </div>
              <button
                onClick={handleClear}
                disabled={disabled}
                className="p-1.5 rounded-lg bg-surface-900/80 text-ink-400 hover:text-red-400 hover:bg-red-950/40 transition-colors duration-150"
                aria-label="Remove selected spectral file"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-4 h-4">
                  <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
                </svg>
              </button>
            </div>
          )}

          {/* File Meta Footer */}
          <div className="px-4 py-2.5 flex items-center justify-between border-t border-olive-subtle bg-surface-900/50">
            <span className="text-[11px] text-ink-400 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-olive-400 inline-block" />
              File ready for inference
            </span>
            <span className="text-[10px] text-ink-500 font-mono">{fileMeta.sizeLabel}</span>
          </div>
        </div>
      )}

      {/* Local validation error */}
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
