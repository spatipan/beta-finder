import type { GymId, GymInfo, RegionId, RegionInfo } from '../types'
import { GYMS, REGIONS } from '../types'

interface Props {
  region: RegionId
  onRegionChange: (r: RegionId) => void
  selected: GymId
  onChange: (gym: GymId) => void
}

function RegionTab({ r, active, onClick }: { r: RegionInfo; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`text-xs font-medium px-1 py-0.5 transition-colors whitespace-nowrap border-b-2
        ${active
          ? 'text-stone-700 dark:text-stone-200 border-primary'
          : 'text-stone-400 dark:text-stone-500 border-transparent hover:text-stone-600 dark:hover:text-stone-300'
        }`}
      aria-pressed={active}
    >
      {r.label}
    </button>
  )
}

function GymPill({ gym, selected, onChange }: { gym: GymInfo; selected: GymId; onChange: (id: GymId) => void }) {
  const isActive = selected === gym.id
  return (
    <button
      className="gym-pill"
      style={
        isActive
          ? { backgroundColor: gym.color, color: '#fff', borderColor: gym.color }
          : { color: gym.color, borderColor: gym.color + '40', backgroundColor: gym.color + '10' }
      }
      onClick={() => onChange(gym.id)}
      aria-pressed={isActive}
    >
      {gym.id === 'all' ? 'All' : gym.label.split(' ')[0]}
    </button>
  )
}

export default function GymFilter({ region, onRegionChange, selected, onChange }: Props) {
  // Show "All" pill + gyms in the selected region only
  const visibleGyms = GYMS.filter((g) => g.id === 'all' || g.region === region)

  function handleRegionChange(r: RegionId) {
    onRegionChange(r)
    onChange('all') // reset gym selection when switching region
  }

  return (
    <div className="space-y-2">
      {/* Region tabs — only shown when multiple regions exist */}
      {REGIONS.length > 1 && (
        <div className="flex gap-1 overflow-x-auto scrollbar-hide">
          {REGIONS.map((r) => (
            <RegionTab
              key={r.id}
              r={r}
              active={region === r.id}
              onClick={() => handleRegionChange(r.id)}
            />
          ))}
        </div>
      )}

      {/* Gym pills for selected region */}
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide">
        {visibleGyms.map((gym) => (
          <GymPill key={gym.id} gym={gym} selected={selected} onChange={onChange} />
        ))}
      </div>
    </div>
  )
}
