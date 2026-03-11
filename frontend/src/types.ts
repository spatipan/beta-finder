export type GymId = 'all' | 'alpine' | 'mainwall' | 'progression'
export type SourceType = 'official' | 'tagged' | 'contributor'
export type RegionId = 'cnx' | 'bkk'

export interface RegionInfo {
  id: RegionId
  label: string
}

// Only CNX for now — region row auto-hides when length <= 1.
// Add { id: 'bkk', label: 'Bangkok' } here when BKK gyms are ready.
export const REGIONS: RegionInfo[] = [
  { id: 'cnx', label: 'Chiang Mai' },
]

export interface GymInfo {
  id: GymId
  label: string
  handle: string
  color: string
  region: RegionId
}

export const GYMS: GymInfo[] = [
  { id: 'all',         label: 'All Gyms',             handle: 'all',                 color: '#F59E0B', region: 'cnx' },
  { id: 'alpine',      label: 'Alpine Outpost',        handle: 'the_alpine_outpost',  color: '#4CAF7D', region: 'cnx' },
  { id: 'mainwall',    label: 'Main Wall CNX',         handle: 'mainwallcnx',         color: '#5B8DEE', region: 'cnx' },
  { id: 'progression', label: 'Progression Vertical',  handle: 'progressionvertical', color: '#FF7043', region: 'cnx' },
  // Add BKK gyms here with region: 'bkk'
]

export interface SearchResult {
  rank: number
  reel_id: string
  url: string
  thumbnail_url: string
  keyframe_urls: string[]
  media_type: 'keyframe' | 'image'
  gym: Exclude<GymId, 'all'>
  source_type: SourceType
  score: number
  username: string
  caption: string
  date: string
}

export interface SearchResponse {
  results: SearchResult[]
  query_valid: boolean
  processing_ms: number
}

export interface StatsResponse {
  total: number
  by_gym: Record<string, number>
  by_source: Record<string, number>
  last_updated: string | null
}

export type SearchState = 'idle' | 'uploading' | 'searching' | 'results' | 'not_a_wall' | 'error' | 'no_results'
