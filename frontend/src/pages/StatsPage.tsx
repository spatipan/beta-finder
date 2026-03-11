import { useEffect, useState } from 'react'
import type { StatsResponse } from '../types'
import { GYMS } from '../types'
import { fetchStats, mockStatsResponse } from '../services/api'

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'

function Bar({ label, value, total, color }: { label: string; value: number; total: number; color: string }) {
  const pct = total > 0 ? (value / total) * 100 : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="font-medium">{label}</span>
        <span className="font-mono text-stone-500">{value}</span>
      </div>
      <div className="h-2.5 rounded-full bg-stone-100 dark:bg-stone-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

export default function StatsPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    async function load() {
      try {
        const data = USE_MOCK ? mockStatsResponse() : await fetchStats()
        setStats(data)
      } catch {
        setError(true)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-stone-400">
        <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
        </svg>
      </div>
    )
  }

  if (error || !stats) {
    return (
      <div className="text-center py-8 text-stone-400">
        <p>Could not load stats — backend offline?</p>
      </div>
    )
  }

  const gymColors: Record<string, string> = {
    alpine: '#4CAF7D',
    mainwall: '#5B8DEE',
    progression: '#FF7043',
  }

  const sourceColors: Record<string, string> = {
    official: '#FF7043',
    tagged: '#F59E0B',
    contributor: '#4CAF7D',
  }

  return (
    <div className="space-y-6">
      {/* Total */}
      <div className="card p-6 text-center">
        <p className="text-5xl font-extrabold text-primary font-mono">{stats.total.toLocaleString()}</p>
        <p className="text-stone-500 mt-1">Reels indexed</p>
      </div>

      {/* By gym */}
      <div className="card p-4 space-y-4">
        <p className="font-bold">By gym</p>
        {GYMS.filter((g) => g.id !== 'all').map((gym) => (
          <Bar
            key={gym.id}
            label={gym.label}
            value={stats.by_gym[gym.id] ?? 0}
            total={stats.total}
            color={gymColors[gym.id] ?? '#FF7043'}
          />
        ))}
      </div>

      {/* By source */}
      <div className="card p-4 space-y-4">
        <p className="font-bold">By source</p>
        {[
          { key: 'official', label: '🏟️ Official gym posts' },
          { key: 'tagged', label: '🏷️ Community tagged posts' },
          { key: 'contributor', label: '👤 Contributors' },
        ].map(({ key, label }) => (
          <Bar
            key={key}
            label={label}
            value={stats.by_source[key] ?? 0}
            total={stats.total}
            color={sourceColors[key] ?? '#999'}
          />
        ))}
      </div>

      {stats.last_updated && (
        <p className="text-xs text-center text-stone-400 font-mono">
          Updated {new Date(stats.last_updated).toLocaleString()}
        </p>
      )}
    </div>
  )
}
