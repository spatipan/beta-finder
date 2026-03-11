import { useState } from 'react'
import type { SearchResult } from '../types'
import { GYMS } from '../types'
import MediaThumb from './MediaThumb'

interface Props {
  result: SearchResult
}

function scoreLabel(score: number): string {
  if (score >= 0.85) return '🎯 Strong match'
  if (score >= 0.70) return '✨ Good match'
  if (score >= 0.55) return '👀 Possible match'
  return '🔍 Weak match'
}

function sourceLabel(source: string): string {
  if (source === 'official') return '🏟️ official'
  if (source === 'tagged') return '🏷️ community'
  return '👤 contributor'
}

export default function ResultCard({ result }: Props) {
  const [hovered, setHovered] = useState(false)
  const gymInfo = GYMS.find((g) => g.id === result.gym)!
  const dateStr = new Date(result.date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
  const caption = result.caption.length > 110 ? result.caption.slice(0, 110) + '…' : result.caption

  return (
    <a
      href={result.url}
      target="_blank"
      rel="noopener noreferrer"
      className="card block group cursor-pointer hover:shadow-md transition-shadow"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Media preview — 9:16 */}
      <div className="relative overflow-hidden" style={{ aspectRatio: '9/16' }}>
        <MediaThumb
          reelId={result.reel_id}
          thumbnailUrl={result.thumbnail_url}
          keyframeUrls={result.keyframe_urls}
          gym={result.gym}
          isHovered={hovered}
        />

        {/* Rank badge */}
        <div className="absolute top-2 left-2 bg-black/60 text-white text-xs font-mono px-2 py-0.5 rounded-lg">
          #{result.rank}
        </div>

        {/* Score badge */}
        <div className="absolute top-2 right-2 bg-black/60 text-white text-xs font-mono px-2 py-0.5 rounded-lg">
          {result.score.toFixed(2)}
        </div>
      </div>

      {/* Info panel */}
      <div className="p-3 space-y-1.5">
        {/* Match quality */}
        {/* <p className="text-sm font-semibold text-stone-700 dark:text-stone-200">
          {scoreLabel(result.score)}
        </p> */}

        {/* Gym + source */}
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full text-white"
            style={{ backgroundColor: gymInfo.color }}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-white/70 inline-block" />
            {gymInfo.label}
          </span>
          <span className="text-xs text-stone-400 dark:text-stone-500">{sourceLabel(result.source_type)}</span>
        </div>

        {/* Username + date */}
        <p className="text-xs text-stone-500 dark:text-stone-400 font-mono">
          @{result.username} · {dateStr}
        </p>

        {/* Caption */}
        {caption && (
          <p className="text-xs text-stone-500 dark:text-stone-400 leading-relaxed line-clamp-2">
            {caption}
          </p>
        )}
      </div>
    </a>
  )
}
