export default function AboutPage() {
  return (
    <div className="space-y-5 pb-4">
      <div className="card p-5 space-y-3">
        <h2 className="font-extrabold text-xl">About BetaFinder CNX</h2>
        <p className="text-stone-500 text-sm leading-relaxed">
          A visual beta discovery tool for Chiang Mai bouldering gyms. Upload a photo of a wall and find
          matching Reels from the local climbing community — so you can study the move before your next session.
        </p>
      </div>

      {/* How it works */}
      <div className="card p-5 space-y-3">
        <h3 className="font-bold">How it works</h3>
        <ol className="space-y-2 text-sm text-stone-500 dark:text-stone-400">
          {[
            'Instagram Reels from gyms and tagged community posts are scraped and indexed every 6 hours',
            'Each Reel is split into frames; the clearest wall frames are selected',
            'DINOv2 embeds each frame into a 768-dim vector — capturing visual structure',
            'Your uploaded photo is embedded the same way, then matched via cosine similarity',
            'Top results are returned with flipbook previews of the original Reel',
          ].map((step, i) => (
            <li key={i} className="flex gap-3">
              <span className="font-mono text-primary font-bold min-w-[1.5rem]">{i + 1}.</span>
              <span>{step}</span>
            </li>
          ))}
        </ol>
      </div>

      {/* Gyms */}
      <div className="card p-5 space-y-3">
        <h3 className="font-bold">Supported gyms</h3>
        <div className="space-y-2 text-sm">
          {[
            { name: 'Alpine Outpost', handle: 'the_alpine_outpost', color: '#4CAF7D' },
            { name: 'Main Wall CNX', handle: 'mainwallcnx', color: '#5B8DEE' },
            { name: 'Progression Vertical', handle: 'progressionvertical', color: '#FF7043' },
          ].map((gym) => (
            <a
              key={gym.handle}
              href={`https://www.instagram.com/${gym.handle}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 group"
            >
              <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: gym.color }} />
              <span className="font-semibold">{gym.name}</span>
              <span className="text-stone-400 font-mono text-xs group-hover:text-primary transition-colors">
                @{gym.handle}
              </span>
            </a>
          ))}
        </div>
      </div>

      {/* Links */}
      <div className="card p-5 space-y-3">
        <h3 className="font-bold">Links</h3>
        <div className="space-y-2 text-sm">
          <a
            href="https://www.instagram.com/patipan_poty"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-stone-600 dark:text-stone-400 hover:text-primary transition-colors"
          >
            <span>📸</span>
            <span className="font-mono">@patipan_poty</span>
          </a>
          <a
            href="https://www.instagram.com/climb.with.poom"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-stone-600 dark:text-stone-400 hover:text-primary transition-colors"
          >
            <span>🧗</span>
            <span className="font-mono">@climb.with.poom</span>
          </a>
        </div>
      </div>

      {/* Inspired by */}
      <p className="text-xs text-center text-stone-400">
        Inspired by{' '}
        <a
          href="https://www.instagram.com/thangman22"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-primary transition-colors"
        >
          BetaScan by @thangman22
        </a>
        {' '}(Warat Wongmaneekit)
      </p>
    </div>
  )
}
