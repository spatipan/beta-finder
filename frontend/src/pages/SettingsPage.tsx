interface Props {
  topK: number
  setTopK: (v: number) => void
  darkMode: boolean
  setDarkMode: (v: boolean) => void
}

export default function SettingsPage({ topK, setTopK, darkMode, setDarkMode }: Props) {
  return (
    <div className="space-y-6">
      <h2 className="font-bold text-lg">Settings</h2>

      {/* Top-K slider */}
      <div className="card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <label htmlFor="topk-slider" className="font-semibold">Results per search</label>
          <span className="font-mono text-primary font-bold text-lg">{topK}</span>
        </div>
        <input
          id="topk-slider"
          type="range"
          min={1}
          max={15}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          className="w-full accent-primary"
        />
        <div className="flex justify-between text-xs text-stone-400">
          <span>1</span>
          <span>15</span>
        </div>
      </div>

      {/* Dark mode toggle */}
      <div className="card p-4 flex items-center justify-between">
        <div>
          <p className="font-semibold">Dark mode</p>
          <p className="text-sm text-stone-400">Easy on the eyes at the gym</p>
        </div>
        <button
          role="switch"
          aria-checked={darkMode}
          onClick={() => setDarkMode(!darkMode)}
          className={`relative w-12 h-6 rounded-full transition-colors ${darkMode ? 'bg-primary' : 'bg-stone-200 dark:bg-stone-700'}`}
        >
          <span
            className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white shadow transition-transform ${darkMode ? 'translate-x-6' : ''}`}
          />
        </button>
      </div>

      {/* Photo tips */}
      <div className="card p-4 space-y-3">
        <p className="font-semibold">📸 Tips for better results</p>
        <ul className="space-y-2 text-sm text-stone-500 dark:text-stone-400">
          {[
            'Shoot the wall straight-on, not at an angle',
            'Include the holds and route markings',
            'Avoid people blocking the wall',
            'Good lighting helps — flash if dark',
            'JPG and PNG work best; HEIC is fine too',
          ].map((tip) => (
            <li key={tip} className="flex gap-2">
              <span className="text-primary">•</span>
              <span>{tip}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
