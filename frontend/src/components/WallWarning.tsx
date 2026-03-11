interface Props {
  onReset: () => void
}

export default function WallWarning({ onReset }: Props) {
  return (
    <div className="rounded-2xl border-2 border-amber-300 bg-amber-50 dark:bg-amber-900/20 dark:border-amber-700 p-5 text-center space-y-3">
      <p className="text-3xl">🤔</p>
      <div>
        <p className="font-bold text-amber-800 dark:text-amber-200">That doesn't look like a climbing wall</p>
        <p className="text-sm text-amber-700 dark:text-amber-400 mt-1">
          Try a photo of the wall itself — holds, panels, and routes work best.
        </p>
      </div>
      <button
        onClick={onReset}
        className="text-sm font-semibold text-amber-700 dark:text-amber-300 underline underline-offset-2"
      >
        Try a different photo
      </button>
    </div>
  )
}
