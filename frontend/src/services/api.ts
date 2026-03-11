import type { SearchResponse, StatsResponse, GymId } from '../types'

const BASE = '/api'

export async function searchBeta(
  file: File,
  gym: GymId = 'all',
  topK: number = 5
): Promise<SearchResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('gym', gym)
  form.append('top_k', String(topK))

  const res = await fetch(`${BASE}/search`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch(`${BASE}/stats`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Backend unreachable')
  return res.json()
}

// Fallback mock for dev when backend is not running
export function mockSearchResponse(): SearchResponse {
  return {
    results: [
      {
        rank: 1,
        reel_id: 'mock_001',
        url: 'https://www.instagram.com/reel/mock_001',
        thumbnail_url: '',
        keyframe_urls: [],
        media_type: 'keyframe',
        gym: 'alpine',
        source_type: 'official',
        score: 0.91,
        username: 'the_alpine_outpost',
        caption: 'Green sloper route on the main cave — great movement on this one! 🧗',
        date: '2024-12-10',
      },
      {
        rank: 2,
        reel_id: 'mock_002',
        url: 'https://www.instagram.com/reel/mock_002',
        thumbnail_url: '',
        keyframe_urls: [],
        media_type: 'keyframe',
        gym: 'mainwall',
        source_type: 'tagged',
        score: 0.76,
        username: 'climb.with.poom',
        caption: 'Finally sent this blue 5C after 4 sessions 💪 #bouldering #mainwallcnx',
        date: '2024-11-28',
      },
    ],
    query_valid: true,
    processing_ms: 142,
  }
}

export function mockStatsResponse(): StatsResponse {
  return {
    total: 792,
    by_gym: { alpine: 312, mainwall: 280, progression: 200 },
    by_source: { official: 520, tagged: 240, contributor: 32 },
    last_updated: new Date().toISOString(),
  }
}
