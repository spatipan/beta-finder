import { useState, useCallback } from 'react'
import type { GymId, RegionId, SearchResult, SearchState } from '../types'
import GymFilter from '../components/GymFilter'
import ImageUploader from '../components/ImageUploader'
import ResultCard from '../components/ResultCard'
import WallWarning from '../components/WallWarning'
import { searchBeta, mockSearchResponse } from '../services/api'

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'

// Confidence tiers (PRD score labels, now also used for grouping)
const TIERS = [
  { label: '🎯 Looks like this route!', min: 0.90 },
  { label: '✨ Worth checking',          min: 0.80 },
  { label: '👀 Similar style',           min: 0.70 },
  { label: '🔍 Different but nearby',   min: 0.00 },
] as const

function groupByTier(results: SearchResult[]) {
  return TIERS.map((tier, i) => ({
    label: tier.label,
    items: results.filter((r) => {
      const upperBound = TIERS[i - 1]?.min ?? Infinity
      return r.score >= tier.min && r.score < upperBound
    }),
  })).filter((g) => g.items.length > 0)
}

interface Props {
  topK: number
  region: RegionId
  onRegionChange: (r: RegionId) => void
}

export default function SearchPage({ topK, region, onRegionChange }: Props) {
  const [gym, setGym] = useState<GymId>('all')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [state, setState] = useState<SearchState>('idle')
  const [results, setResults] = useState<SearchResult[]>([])
  const [processingMs, setProcessingMs] = useState<number | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const handleFile = useCallback((f: File) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setState('uploading')
    setResults([])
    setErrorMsg(null)
  }, [])

  const handleSearch = useCallback(async () => {
    if (!file) return
    setState('searching')

    try {
      let data
      if (USE_MOCK) {
        await new Promise((r) => setTimeout(r, 800))
        data = mockSearchResponse()
      } else {
        data = await searchBeta(file, gym, topK)
      }

      setProcessingMs(data.processing_ms)

      if (!data.query_valid) {
        setState('not_a_wall')
        return
      }
      if (data.results.length === 0) {
        setState('no_results')
        return
      }
      setResults(data.results)
      setState('results')
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : 'Unknown error')
      setState('error')
    }
  }, [file, gym, topK])

  const handleReset = useCallback(() => {
    setFile(null)
    setPreview(null)
    setState('idle')
    setResults([])
    setErrorMsg(null)
  }, [])

  const handleGymChange = useCallback((g: GymId) => {
    setGym(g)
    setResults([])
    setState(file ? 'uploading' : 'idle')
  }, [file])

  return (
    <div className="space-y-4">
      {/* Gym filter (with region row) */}
      <GymFilter
        region={region}
        onRegionChange={onRegionChange}
        selected={gym}
        onChange={handleGymChange}
      />

      {/* Uploader */}
      <ImageUploader
        onFile={handleFile}
        preview={preview}
        disabled={state === 'searching'}
      />

      {/* Find Beta button */}
      {(state === 'uploading' || state === 'results' || state === 'no_results') && (
        <button
          className="btn-primary w-full text-base"
          onClick={handleSearch}
          disabled={state === 'searching'}
        >
          🔍 Find Beta
        </button>
      )}

      {/* Searching spinner */}
      {state === 'searching' && (
        <button className="btn-primary w-full text-base" disabled>
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
            Looking for beta…
          </span>
        </button>
      )}

      {/* Not a wall warning */}
      {state === 'not_a_wall' && <WallWarning onReset={handleReset} />}

      {/* Error */}
      {state === 'error' && (
        <div className="rounded-2xl border-2 border-red-200 bg-red-50 dark:bg-red-900/20 dark:border-red-800 p-4 space-y-2">
          <p className="font-semibold text-red-700 dark:text-red-300">⚡ Something went wrong</p>
          <p className="text-sm text-red-600 dark:text-red-400">{errorMsg}</p>
          <p className="text-xs text-red-500">Make sure the backend is running: <code className="font-mono">uvicorn backend.app.main:app</code></p>
          <button onClick={handleSearch} className="text-sm font-semibold text-red-600 underline">Retry</button>
        </div>
      )}

      {/* No results */}
      {state === 'no_results' && (
        <div className="text-center py-8 text-stone-400 space-y-2">
          <p className="text-3xl">🪨</p>
          <p className="font-semibold">No beta found</p>
          <p className="text-sm">Try searching "All Gyms" or upload a different photo</p>
        </div>
      )}

      {/* Results — grouped by confidence tier */}
      {state === 'results' && results.length > 0 && (
        <div className="space-y-5">
          {/* Summary row */}
          <div className="flex items-center justify-between text-xs text-stone-400 dark:text-stone-500">
            <span>{results.length} result{results.length !== 1 ? 's' : ''}</span>
            {processingMs !== null && <span className="font-mono">{processingMs}ms</span>}
          </div>

          {/* Tier groups */}
          {groupByTier(results).map((group) => (
            <div key={group.label} className="space-y-2">
              <h3 className="text-xs font-bold text-stone-400 dark:text-stone-500 uppercase tracking-wider px-0.5">
                {group.label}
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {group.items.map((r) => (
                  <ResultCard key={r.reel_id} result={r} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
