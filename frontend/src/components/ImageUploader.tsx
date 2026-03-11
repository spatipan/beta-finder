import { useRef, useState } from 'react'

interface Props {
  onFile: (file: File) => void
  preview: string | null
  disabled?: boolean
}

export default function ImageUploader({ onFile, preview, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  // Default portrait (3:4) — most wall photos are taken vertically on a phone
  const [aspectRatio, setAspectRatio] = useState<'3/4' | '3/2'>('3/4')

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return
    const file = files[0]
    if (!file.type.match(/^image\//)) return

    // Detect orientation — switch container to landscape only for wide images
    const url = URL.createObjectURL(file)
    const img = new window.Image()
    img.onload = () => {
      setAspectRatio(img.naturalWidth > img.naturalHeight ? '3/2' : '3/4')
      URL.revokeObjectURL(url)
    }
    img.src = url

    onFile(file)
  }

  return (
    <div
      className={`relative rounded-2xl border-2 border-dashed transition-all cursor-pointer
        ${dragging ? 'border-primary bg-primary/5 scale-[1.01]' : 'border-stone-200 dark:border-stone-700 hover:border-primary/50'}
        ${disabled ? 'opacity-50 pointer-events-none' : ''}
      `}
      style={{ aspectRatio }}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files) }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/heic,image/webp"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
        disabled={disabled}
      />

      {preview ? (
        <img
          src={preview}
          alt="Uploaded wall"
          className="w-full h-full object-cover rounded-2xl"
        />
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-stone-400 dark:text-stone-500">
          <span className="text-4xl">🧗</span>
          <div className="text-center">
            <p className="font-semibold text-stone-600 dark:text-stone-300">Drop a wall photo</p>
            <p className="text-sm mt-0.5">or tap to browse — JPG, PNG, HEIC</p>
          </div>
        </div>
      )}

      {preview && (
        <div className="absolute top-2 right-2">
          <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-lg font-mono">
            tap to change
          </span>
        </div>
      )}
    </div>
  )
}
