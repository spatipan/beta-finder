import { useState } from 'react'
import type { GymId } from '../types'
import { GYMS } from '../types'

interface Props {
  reelId: string
  thumbnailUrl: string
  keyframeUrls: string[]
  gym: Exclude<GymId, 'all'>
  isHovered: boolean
}

// Warm placeholder when no cached images exist
function WallPlaceholder({ gym }: { gym: Exclude<GymId, 'all'> }) {
  const gymInfo = GYMS.find((g) => g.id === gym)!
  const color = gymInfo.color

  // Deterministic hold positions per gym
  const holds = [
    { x: 25, y: 20 }, { x: 70, y: 35 }, { x: 45, y: 55 },
    { x: 20, y: 70 }, { x: 75, y: 65 }, { x: 55, y: 80 },
    { x: 35, y: 40 }, { x: 80, y: 20 }, { x: 60, y: 50 },
  ]

  return (
    <div className="w-full h-full relative overflow-hidden" style={{ background: '#EDE8E2' }}>
      {/* Grid lines */}
      <svg className="absolute inset-0 w-full h-full opacity-20" preserveAspectRatio="none">
        <defs>
          <pattern id={`grid-${gym}`} width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#9C8B7A" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill={`url(#grid-${gym})`} />
      </svg>

      {/* Hold dots */}
      {holds.map((h, i) => (
        <div
          key={i}
          className="absolute rounded-full opacity-70"
          style={{
            left: `${h.x}%`,
            top: `${h.y}%`,
            width: i % 3 === 0 ? 14 : 10,
            height: i % 3 === 0 ? 14 : 10,
            backgroundColor: i % 2 === 0 ? color : '#D4C4B0',
            transform: 'translate(-50%, -50%)',
            boxShadow: `0 2px 4px ${color}40`,
          }}
        />
      ))}

      <div className="absolute bottom-2 left-2">
        <span
          className="text-xs font-mono px-1.5 py-0.5 rounded-md text-white"
          style={{ backgroundColor: color + 'CC' }}
        >
          {gymInfo.label.split(' ')[0]}
        </span>
      </div>
    </div>
  )
}

export default function MediaThumb({ reelId, thumbnailUrl, keyframeUrls, gym, isHovered }: Props) {
  const [imgError, setImgError] = useState(false)

  const hasImages = thumbnailUrl && !imgError

  if (!hasImages) {
    return <WallPlaceholder gym={gym} />
  }

  return (
    <div className="w-full h-full relative">
      {/* Static thumbnail (shown when not hovered) */}
      <img
        src={thumbnailUrl}
        alt={`Reel ${reelId} thumbnail`}
        className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-200 ${isHovered && keyframeUrls.length > 0 ? 'opacity-0' : 'opacity-100'}`}
        onError={() => setImgError(true)}
      />

      {/* Flipbook frames (shown on hover) */}
      {isHovered && keyframeUrls.length > 0 && (
        <div className="absolute inset-0">
          {keyframeUrls.map((url, i) => (
            <img
              key={i}
              src={url}
              alt={`Frame ${i}`}
              className="flipbook-frame"
              style={{ animationDelay: `${i * 0.5}s` }}
            />
          ))}
        </div>
      )}
    </div>
  )
}
