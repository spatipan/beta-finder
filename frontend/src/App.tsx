import { useState, useEffect } from 'react'
import type { RegionId } from './types'
import SearchPage from './pages/SearchPage'
import SettingsPage from './pages/SettingsPage'
import StatsPage from './pages/StatsPage'
import AboutPage from './pages/AboutPage'

type Tab = 'search' | 'settings' | 'stats' | 'about'

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'search', label: 'Search', icon: '🔍' },
  { id: 'settings', label: 'Settings', icon: '⚙️' },
  { id: 'stats', label: 'Stats', icon: '📊' },
  { id: 'about', label: 'About', icon: 'ℹ️' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('search')
  const [topK, setTopK] = useState(5)
  const [darkMode, setDarkMode] = useState(false)
  const [region, setRegion] = useState<RegionId>('cnx')

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
  }, [darkMode])

  return (
    <div className="min-h-dvh bg-chalk dark:bg-surface-dark flex justify-center">
      <div className="w-full max-w-app flex flex-col min-h-dvh relative">
        {/* Header */}
        <header className="sticky top-0 z-20 bg-chalk/90 dark:bg-surface-dark/90 backdrop-blur-sm border-b border-stone-100 dark:border-stone-800 px-4 py-3">
          <div className="flex items-baseline gap-2">
            <h1 className="font-extrabold text-2xl tracking-tight">
              <span className="text-stone-800 dark:text-stone-100">Beta</span>
              <span className="text-primary">Finder</span>
            </h1>
            <span className="font-mono text-stone-400 dark:text-stone-500 text-sm">CNX</span>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 px-4 py-4 overflow-y-auto pb-20">
          {tab === 'search' && <SearchPage topK={topK} region={region} onRegionChange={setRegion} />}
          {tab === 'settings' && (
            <SettingsPage
              topK={topK}
              setTopK={setTopK}
              darkMode={darkMode}
              setDarkMode={setDarkMode}
            />
          )}
          {tab === 'stats' && <StatsPage />}
          {tab === 'about' && <AboutPage />}
        </main>

        {/* Bottom tab bar */}
        <nav className="fixed bottom-0 left-1/2 -translate-x-1/2 w-full max-w-app bg-white/95 dark:bg-stone-900/95 backdrop-blur-sm border-t border-stone-100 dark:border-stone-800 flex z-20">
          {TABS.map(({ id, label, icon }) => (
            <button
              key={id}
              className={`tab-item ${tab === id ? 'active' : ''}`}
              onClick={() => setTab(id)}
            >
              <span className="text-xl leading-none">{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </div>
    </div>
  )
}
