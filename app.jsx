import { useState, useRef, useCallback, useEffect } from "react";

const API_BASE = "http://localhost:8000";

// ─────────────────────────────────────────────────────────────────────────────
// DESIGN SYSTEM
// Vibe: easy-going curiosity → discovery → small joy of finding the beta
// Typography: Nunito (rounded, friendly) + JetBrains Mono (scores only)
// Light default / dark toggle
// ─────────────────────────────────────────────────────────────────────────────

const LIGHT = {
  bg: "#f7f5f2",   // warm off-white — chalk dust on a wooden floor
  bgCard: "#ffffff",
  bgSubtle: "#f0ede8",
  bgInput: "#ede9e3",
  border: "#e0dbd3",

  text: "#2c2420",
  textSub: "#7a6f66",
  textMuted: "#b0a89e",

  orange: "#ff7043",   // terracotta — primary, curiosity
  amber: "#f59e0b",   // golden hour
  green: "#4caf7d",   // send color 🎉
  teal: "#2ab5a6",
  blue: "#5b8dee",

  gymAlpine: "#4caf7d",
  gymMain: "#5b8dee",
  gymProg: "#ff7043",
};

const DARK = {
  bg: "#1c1917",
  bgCard: "#27231f",
  bgSubtle: "#2e2925",
  bgInput: "#332e29",
  border: "#3d3830",

  text: "#f5f0eb",
  textSub: "#a89e94",
  textMuted: "#5c5550",

  orange: "#ff7043",
  amber: "#f59e0b",
  green: "#4caf7d",
  teal: "#2ab5a6",
  blue: "#5b8dee",

  gymAlpine: "#4caf7d",
  gymMain: "#5b8dee",
  gymProg: "#ff7043",
};

const GYMS = [
  { key: "all", label: "All", full: "All Gyms", gc: t => t.amber },
  { key: "alpine", label: "Alpine", full: "Alpine Outpost", gc: t => t.gymAlpine },
  { key: "mainwall", label: "Main Wall", full: "Main Wall CNX", gc: t => t.gymMain },
  { key: "progression", label: "Progression", full: "Progression Vertical", gc: t => t.gymProg },
];

const SOURCE_META = {
  tagged: { label: "community", emoji: "🏷️" },
  official: { label: "official", emoji: "🏟️" },
  contributor: { label: "contributor", emoji: "👤" },
};


// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────
const scoreEmoji = s => s >= 0.90 ? "🎯" : s >= 0.80 ? "✨" : s >= 0.70 ? "👀" : "🔍";
const scoreWord = s => s >= 0.90 ? "Strong match" : s >= 0.80 ? "Good match" : s >= 0.70 ? "Possible match" : "Weak match";
const gymCfg = k => GYMS.find(g => g.key === k) || GYMS[0];

// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL STYLES
// ─────────────────────────────────────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Nunito', sans-serif; -webkit-font-smoothing: antialiased; }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: #d0cbc4; border-radius: 4px; }

  @keyframes slideUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
  }
  .su  { animation: slideUp .35s ease both; }
  .su1 { animation: slideUp .35s .05s ease both; }
  .su2 { animation: slideUp .35s .10s ease both; }
  .su3 { animation: slideUp .35s .15s ease both; }
  .su4 { animation: slideUp .35s .20s ease both; }
  .su5 { animation: slideUp .35s .25s ease both; }

  /* Card hover */
  .rc { transition: box-shadow .2s, transform .2s; }
  .rc:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.10) !important; }

  /* Flipbook — fires on .rc:hover */
  @keyframes f0 { 0%,24%,100%{opacity:1} 25%,99%{opacity:0} }
  @keyframes f1 { 0%,24%{opacity:0} 25%,49%,100%{opacity:1} 50%,99%{opacity:0} }
  @keyframes f2 { 0%,49%{opacity:0} 50%,74%,100%{opacity:1} 75%,99%{opacity:0} }
  @keyframes f3 { 0%,74%{opacity:0} 75%,100%{opacity:1} }
  .fr0 { opacity:1; }
  .fr1,.fr2,.fr3 { opacity:0; position:absolute; inset:0; width:100%; height:100%; }
  .rc:hover .fr0 { animation: f0 2.4s steps(1) infinite; }
  .rc:hover .fr1 { animation: f1 2.4s steps(1) infinite; }
  .rc:hover .fr2 { animation: f2 2.4s steps(1) infinite; }
  .rc:hover .fr3 { animation: f3 2.4s steps(1) infinite; }

  @keyframes spin { to { transform:rotate(360deg); } }
  .spinner { animation: spin .8s linear infinite; }

  .sbtn { transition: all .2s; }
  .sbtn:not(:disabled):hover { transform: translateY(-2px); }
  .sbtn:not(:disabled):active { transform: translateY(0); }

  .gpill { transition: all .15s; }
  .gpill:hover { opacity: .85; transform: translateY(-1px); }

  input[type=range] { cursor: pointer; }
`;

// ─────────────────────────────────────────────────────────────────────────────
// SHARED COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────
function SectionLabel({ children, t }) {
  return (
    <p style={{
      fontWeight: 700, fontSize: 11, letterSpacing: "1px",
      textTransform: "uppercase", color: t.textMuted, marginBottom: 10,
    }}>{children}</p>
  );
}

function Pill({ children, active, color, onClick, t }) {
  return (
    <button className="gpill" onClick={onClick} style={{
      padding: "7px 16px", borderRadius: 99,
      border: `1.5px solid ${active ? color : t.border}`,
      background: active ? color + "18" : "transparent",
      color: active ? color : t.textSub,
      fontFamily: "'Nunito',sans-serif",
      fontWeight: active ? 700 : 500,
      fontSize: 14, cursor: "pointer", lineHeight: 1,
    }}>{children}</button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MEDIA THUMBNAIL (9:16 vertical — matches Reel format)
// thumbnailUrl / keyframeUrls are null in prototype.
// In production: swap with /api/thumb/{shortcode} and /api/frames/{shortcode}/n
// ─────────────────────────────────────────────────────────────────────────────
function MediaThumb({ result, gymColor }) {
  const isReel = result.mediaType === "keyframe" && result.keyframeUrls?.length;
  const count = isReel ? result.keyframeUrls.length : 1;

  // Warm placeholder palette per gym
  const BG = {
    alpine: ["#1b3628", "#1f4030", "#173325", "#1b3628"],
    mainwall: ["#1b2c3a", "#1f3245", "#172535", "#1b2c3a"],
    progression: ["#3a2618", "#45301d", "#352215", "#3a2618"],
    all: ["#2d2820", "#332e26", "#28231a", "#2d2820"],
  };
  const bgs = BG[result.gym] || BG.all;

  const dots = [
    { l: "22%", t: "30%", s: 10 }, { l: "58%", t: "20%", s: 8 }, { l: "74%", t: "48%", s: 11 },
    { l: "35%", t: "64%", s: 9 }, { l: "50%", t: "76%", s: 8 }, { l: "16%", t: "56%", s: 7 },
  ];

  return (
    <div style={{
      width: "100%", aspectRatio: "9/16",
      borderRadius: "12px 12px 0 0",
      overflow: "hidden", position: "relative",
      background: "#111", flexShrink: 0,
    }}>
      {Array.from({ length: count }).map((_, i) => {
        const rawSrc = isReel ? result.keyframeUrls[i] : (result.thumbnailUrl ?? null);
        const src = rawSrc?.startsWith("/api/") ? `${API_BASE}${rawSrc}` : rawSrc;
        return (
          <div key={i} className={`fr${i}`} style={{
            width: "100%", height: "100%",
            position: i === 0 ? "relative" : "absolute",
            top: 0, left: 0,
          }}>
            {src
              ? <img src={src} alt="" style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
              : (
                <div style={{ width: "100%", height: "100%", background: bgs[i % bgs.length], position: "relative" }}>
                  {/* Subtle wall grid */}
                  <div style={{
                    position: "absolute", inset: 0,
                    backgroundImage: `linear-gradient(rgba(255,255,255,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.025) 1px,transparent 1px)`,
                    backgroundSize: "28px 28px",
                  }} />
                  {/* Hold dots */}
                  {dots.map((d, di) => (
                    <div key={di} style={{
                      position: "absolute", left: d.l, top: d.t,
                      width: d.s, height: d.s,
                      borderRadius: "35% 65% 65% 35% / 35% 35% 65% 65%",
                      background: gymColor, opacity: 0.65 + di * 0.04,
                      boxShadow: `0 2px 8px ${gymColor}55`,
                      transform: "translate(-50%,-50%)",
                    }} />
                  ))}
                </div>
              )
            }
          </div>
        );
      })}

      {/* Bottom fade */}
      <div style={{
        position: "absolute", inset: 0,
        background: "linear-gradient(to top, rgba(0,0,0,0.5) 0%, transparent 45%)",
        pointerEvents: "none",
      }} />

      {/* Media badge top-left */}
      <div style={{
        position: "absolute", top: 10, left: 10,
        background: "rgba(0,0,0,0.45)", backdropFilter: "blur(6px)",
        borderRadius: 6, padding: "3px 8px",
        display: "flex", alignItems: "center", gap: 4,
      }}>
        <span style={{ fontSize: 11 }}>{isReel ? "🎬" : "🖼️"}</span>
        <span style={{
          fontFamily: "'JetBrains Mono',monospace",
          fontSize: 9, color: "rgba(255,255,255,0.8)", letterSpacing: "0.5px",
        }}>{isReel ? "REEL" : "PHOTO"}</span>
      </div>

      {/* Hover hint (Reels only) */}
      {isReel && (
        <div style={{
          position: "absolute", top: 10, right: 10,
          background: "rgba(0,0,0,0.35)", backdropFilter: "blur(6px)",
          borderRadius: 6, padding: "3px 8px",
        }}>
          <span style={{
            fontFamily: "'JetBrains Mono',monospace",
            fontSize: 9, color: "rgba(255,255,255,0.4)",
          }}>hover to preview</span>
        </div>
      )}

      {/* Frame dots */}
      {isReel && (
        <div style={{
          position: "absolute", bottom: 10, left: "50%", transform: "translateX(-50%)",
          display: "flex", gap: 4,
        }}>
          {Array.from({ length: count }).map((_, i) => (
            <div key={i} style={{
              width: 4, height: 4, borderRadius: "50%",
              background: i === 0 ? gymColor : "rgba(255,255,255,0.3)",
            }} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RESULT CARD — vertical, click whole card → IG
// No wall% shown. Score is human-readable word + emoji.
// ─────────────────────────────────────────────────────────────────────────────
function ResultCard({ result, index, t }) {
  const gym = gymCfg(result.gym);
  const color = gym.gc(t);
  const pct = Math.round(result.score * 100);
  const src = SOURCE_META[result.sourceType] || SOURCE_META.tagged;

  return (
    <a
      href={result.url} target="_blank" rel="noopener noreferrer"
      className={`rc su${Math.min(index + 1, 5)}`}
      style={{
        display: "block", textDecoration: "none",
        background: t.bgCard, borderRadius: 16,
        border: `1.5px solid ${t.border}`,
        overflow: "hidden",
        boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
      }}
    >
      {/* 9:16 media preview */}
      <MediaThumb result={result} gymColor={color} />

      {/* Info section */}
      <div style={{ padding: "14px 14px 16px" }}>

        {/* Gym + source */}
        <div style={{
          display: "flex", alignItems: "center",
          justifyContent: "space-between", marginBottom: 8,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: color, flexShrink: 0,
            }} />
            <span style={{ fontWeight: 700, fontSize: 13, color }}>{gym.full}</span>
          </div>
          <span style={{ fontSize: 11, color: t.textMuted }}>
            {src.emoji} {src.label}
          </span>
        </div>

        {/* Score chip — friendly, not a bar */}
        <div style={{
          display: "inline-flex", alignItems: "center", gap: 5,
          background: color + "14", borderRadius: 8,
          padding: "4px 10px", marginBottom: 10,
        }}>
          <span style={{ fontSize: 14 }}>{scoreEmoji(result.score)}</span>
          <span style={{ fontWeight: 700, fontSize: 13, color }}>{scoreWord(result.score)}</span>
          <span style={{
            fontFamily: "'JetBrains Mono',monospace",
            fontSize: 11, color: color + "bb",
          }}>{pct}%</span>
        </div>

        {/* Username + date */}
        <div style={{ fontWeight: 700, fontSize: 13, color: t.text, marginBottom: 5 }}>
          @{result.username}
          <span style={{ fontWeight: 400, fontSize: 11, color: t.textMuted, marginLeft: 6 }}>
            {result.date}
          </span>
        </div>

        {/* Caption */}
        <p style={{ fontSize: 13, color: t.textSub, lineHeight: 1.6, margin: 0 }}>
          {result.caption.length > 110 ? result.caption.slice(0, 110) + "…" : result.caption}
        </p>

        {/* Tap hint */}
        <div style={{
          marginTop: 10, fontSize: 11, color: t.textMuted,
          display: "flex", alignItems: "center", gap: 4,
        }}>
          <span style={{ color }}>↗</span>
          <span>Open on Instagram</span>
        </div>
      </div>
    </a>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────────────────
export default function BetaFinder() {
  const [dark, setDark] = useState(false);
  const [tab, setTab] = useState("search");
  const [gym, setGym] = useState("all");
  const [model, setModel] = useState("ViT-B-32");
  const [topK, setTopK] = useState(5);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const fileRef = useRef();
  const t = dark ? DARK : LIGHT;

  // Load stats once on mount (and when switching to stats tab)
  useEffect(() => {
    if (tab !== "stats") return;
    fetch(`${API_BASE}/api/stats`)
      .then(r => r.json())
      .then(setStats)
      .catch(() => setStats(null));
  }, [tab]);

  const handleFile = useCallback(f => {
    if (!f?.type.startsWith("image/")) return;
    setFile(f); setPreview(URL.createObjectURL(f));
    setResults([]); setSearched(false); setError(null);
  }, []);

  const handleSearch = async () => {
    if (!file) return;
    setLoading(true); setResults([]); setError(null);

    const form = new FormData();
    form.append("file", file);
    form.append("gym", gym);
    form.append("topK", topK);
    form.append("model", model);

    try {
      const res = await fetch(`${API_BASE}/api/search`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      setResults(data.results || []);
    } catch (e) {
      setError(e.message || "Search failed");
    } finally {
      setLoading(false); setSearched(true);
    }
  };

  const gymColor = gymCfg(gym).gc(t);

  return (
    <>
      <style>{CSS}</style>
      <div style={{
        minHeight: "100vh", background: t.bg, color: t.text,
        fontFamily: "'Nunito',sans-serif",
        transition: "background .25s, color .25s",
      }}>
        <div style={{ maxWidth: 440, margin: "0 auto", padding: "0 18px 80px" }}>

          {/* ── Header ──────────────────────────────────────────────── */}
          <header className="su" style={{
            paddingTop: 36, paddingBottom: 20,
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <div>
              <div style={{ fontWeight: 900, fontSize: 26, lineHeight: 1, letterSpacing: "-0.5px" }}>
                Beta<span style={{ color: t.orange }}>Finder</span>
                <span style={{
                  fontFamily: "'JetBrains Mono',monospace",
                  fontSize: 11, fontWeight: 400, color: t.textMuted, marginLeft: 6,
                }}>CNX</span>
              </div>
              <div style={{ fontSize: 12, color: t.textSub, marginTop: 3, fontWeight: 600 }}>
                หา beta · เชียงใหม่ climbing 🧗
              </div>
            </div>

            {/* Light/dark toggle */}
            <button onClick={() => setDark(d => !d)} style={{
              width: 46, height: 26, borderRadius: 13, border: "none",
              background: dark ? t.orange : t.bgInput,
              cursor: "pointer", position: "relative",
              transition: "background .2s", flexShrink: 0,
            }}>
              <div style={{
                position: "absolute",
                top: 3, left: dark ? "calc(100% - 23px)" : 3,
                width: 20, height: 20, borderRadius: "50%",
                background: "#fff",
                boxShadow: "0 1px 4px rgba(0,0,0,0.2)",
                transition: "left .2s",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11,
              }}>{dark ? "🌙" : "☀️"}</div>
            </button>
          </header>

          {/* ── Tabs ────────────────────────────────────────────────── */}
          <nav className="su1" style={{ display: "flex", gap: 4, marginBottom: 24 }}>
            {[
              { key: "search", label: "Search" },
              { key: "settings", label: "Settings" },
              { key: "stats", label: "Stats" },
              { key: "about", label: "About" },
            ].map(tb => (
              <button key={tb.key} onClick={() => setTab(tb.key)} style={{
                flex: 1, padding: "9px 4px", borderRadius: 10, border: "none",
                background: tab === tb.key ? t.bgCard : "transparent",
                boxShadow: tab === tb.key ? "0 1px 6px rgba(0,0,0,0.08)" : "none",
                color: tab === tb.key ? t.orange : t.textSub,
                fontFamily: "'Nunito',sans-serif",
                fontWeight: tab === tb.key ? 700 : 500,
                fontSize: 13, cursor: "pointer",
                transition: "all .15s",
              }}>{tb.label}</button>
            ))}
          </nav>

          {/* ══════════════ SEARCH ══════════════ */}
          {tab === "search" && <>
            {/* Gym selector */}
            <div className="su1" style={{ marginBottom: 20 }}>
              <SectionLabel t={t}>Gym</SectionLabel>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {GYMS.map(g => (
                  <Pill key={g.key} active={gym === g.key} color={g.gc(t)}
                    onClick={() => setGym(g.key)} t={t}>
                    {g.label}
                  </Pill>
                ))}
              </div>
            </div>

            {/* Upload */}
            <div className="su2" style={{ marginBottom: 16 }}>
              <SectionLabel t={t}>Wall Photo</SectionLabel>
              <div
                onDragOver={e => { e.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files?.[0]) }}
                onClick={() => fileRef.current?.click()}
                style={{
                  borderRadius: 16,
                  border: `2px dashed ${dragging ? t.orange : t.border}`,
                  background: dragging ? t.orange + "08" : t.bgSubtle,
                  overflow: "hidden", cursor: "pointer",
                  transition: "border-color .15s, background .15s",
                  minHeight: preview ? "auto" : 160,
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}
              >
                <input ref={fileRef} type="file" accept="image/*,.heic"
                  style={{ display: "none" }}
                  onChange={e => handleFile(e.target.files?.[0])}
                />
                {preview ? (
                  <div style={{ position: "relative", width: "100%" }}>
                    <img src={preview} alt="" style={{
                      width: "100%", maxHeight: 280, objectFit: "cover", display: "block",
                    }} />
                    <div style={{
                      position: "absolute", inset: 0,
                      background: "linear-gradient(to top, rgba(0,0,0,0.4), transparent)",
                      display: "flex", alignItems: "flex-end", padding: "10px 12px",
                    }}>
                      <span style={{ fontSize: 11, color: "rgba(255,255,255,0.75)" }}>
                        {file?.name} · tap to change
                      </span>
                    </div>
                  </div>
                ) : (
                  <div style={{ textAlign: "center", padding: 32 }}>
                    <div style={{ fontSize: 36, marginBottom: 10, opacity: 0.4 }}>📷</div>
                    <div style={{ fontSize: 14, color: t.textSub, fontWeight: 600, marginBottom: 4 }}>
                      Drop your wall photo here
                    </div>
                    <div style={{ fontSize: 12, color: t.textMuted }}>JPG · PNG · HEIC</div>
                  </div>
                )}
              </div>
            </div>

            {/* Search button */}
            <div className="su3" style={{ marginBottom: 28 }}>
              <button
                className="sbtn" onClick={handleSearch}
                disabled={!file || loading}
                style={{
                  width: "100%", padding: "15px", borderRadius: 14, border: "none",
                  background: file && !loading
                    ? `linear-gradient(135deg, ${t.orange}, #ff8c42)`
                    : t.bgInput,
                  color: file && !loading ? "#fff" : t.textMuted,
                  fontFamily: "'Nunito',sans-serif",
                  fontWeight: 800, fontSize: 15,
                  cursor: file && !loading ? "pointer" : "not-allowed",
                  boxShadow: file && !loading ? `0 4px 20px ${t.orange}44` : "none",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                }}
              >
                {loading ? (
                  <>
                    <div className="spinner" style={{
                      width: 16, height: 16,
                      border: "2px solid rgba(255,255,255,0.3)",
                      borderTopColor: "#fff", borderRadius: "50%",
                    }} />
                    Looking for beta…
                  </>
                ) : <>🔍 Find Beta</>}
              </button>
            </div>

            {/* Results */}
            {results.length > 0 && (
              <div>
                <div className="su" style={{
                  display: "flex", justifyContent: "space-between",
                  alignItems: "baseline", marginBottom: 14,
                }}>
                  <SectionLabel t={t}>{results.length} matches found ✨</SectionLabel>
                  <span style={{ fontSize: 11, color: t.textMuted }}>tap card → Instagram</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  {results.map((r, i) => (
                    <ResultCard key={r.rank} result={r} index={i} t={t} />
                  ))}
                </div>
              </div>
            )}

            {error && (
              <div style={{ textAlign: "center", padding: 40, color: t.textMuted }}>
                <div style={{ fontSize: 40, marginBottom: 10 }}>⚠️</div>
                <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4, color: t.textSub }}>
                  {error}
                </div>
                <div style={{ fontSize: 12 }}>Make sure the API server is running: <code>uvicorn api:app --reload</code></div>
              </div>
            )}

            {searched && !error && results.length === 0 && (
              <div style={{ textAlign: "center", padding: 40, color: t.textMuted }}>
                <div style={{ fontSize: 40, marginBottom: 10 }}>🤔</div>
                <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4, color: t.textSub }}>
                  No matches yet
                </div>
                <div style={{ fontSize: 12 }}>
                  Try "All" gyms or add more contributors!
                </div>
              </div>
            )}
          </>}

          {/* ══════════════ SETTINGS ══════════════ */}
          {tab === "settings" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {/* Model */}
              {/* <div className="su" style={{
                background: t.bgCard, borderRadius: 16,
                border: `1.5px solid ${t.border}`, overflow: "hidden",
                boxShadow: "0 1px 8px rgba(0,0,0,0.05)",
              }}> */}
              {/* <div style={{
                  padding: "12px 16px", borderBottom: `1px solid ${t.border}`,
                  fontWeight: 700, fontSize: 13, color: t.textSub,
                }}>🧠 CLIP Model</div> */}
              {/* {[
                  {key:"ViT-B-32",name:"ViT-B/32",desc:"Fast — good for most walls",   badge:"⚡ Fast"},
                  {key:"ViT-L-14",name:"ViT-L/14",desc:"More accurate, slower to run", badge:"🎯 Accurate"},
                ].map((m,i) => (
                  <div key={m.key} onClick={() => setModel(m.key)} style={{
                    padding:"14px 16px",display:"flex",alignItems:"center",gap:12,
                    cursor:"pointer",
                    borderBottom: i===0 ? `1px solid ${t.border}` : "none",
                    background: model===m.key ? t.orange+"0d" : "transparent",
                    transition:"background .15s",
                  }}>
                    <div style={{
                      width:20,height:20,borderRadius:"50%",
                      border:`2px solid ${model===m.key ? t.orange : t.border}`,
                      background: model===m.key ? t.orange : "transparent",
                      display:"flex",alignItems:"center",justifyContent:"center",
                      flexShrink:0,transition:"all .15s",
                    }}>
                      {model===m.key && <div style={{width:6,height:6,borderRadius:"50%",background:"#fff"}}/>}
                    </div>
                    <div style={{flex:1}}>
                      <div style={{fontWeight:700,fontSize:14,color: model===m.key ? t.orange : t.text}}>
                        {m.name}
                      </div>
                      <div style={{fontSize:12,color:t.textMuted,marginTop:1}}>{m.desc}</div>
                    </div>
                    <span style={{fontSize:11,fontWeight:600,color: model===m.key ? t.orange : t.textMuted}}>
                      {m.badge}
                    </span>
                  </div>
                ))} */}
              {/* </div> */}

              {/* Top-K */}
              <div className="su1" style={{
                background: t.bgCard, borderRadius: 16,
                border: `1.5px solid ${t.border}`, padding: "16px",
                boxShadow: "0 1px 8px rgba(0,0,0,0.05)",
              }}>
                <div style={{
                  display: "flex", justifyContent: "space-between",
                  alignItems: "center", marginBottom: 12,
                }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: t.textSub }}>
                    🎯 Results to show
                  </div>
                  <span style={{ fontWeight: 900, fontSize: 22, color: t.orange, lineHeight: 1 }}>
                    {topK}
                  </span>
                </div>
                <input type="range" min={1} max={15} value={topK}
                  onChange={e => setTopK(Number(e.target.value))}
                  style={{
                    width: "100%", height: 4, appearance: "none",
                    borderRadius: 2, outline: "none",
                    background: `linear-gradient(to right, ${t.orange} ${(topK / 15) * 100}%, ${t.bgInput} ${(topK / 15) * 100}%)`,
                  }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: 11, color: t.textMuted }}>
                  <span>1</span><span>15</span>
                </div>
              </div>

              {/* Tips */}
              <div className="su2" style={{
                background: t.bgCard, borderRadius: 16,
                border: `1.5px solid ${t.border}`, overflow: "hidden",
                boxShadow: "0 1px 8px rgba(0,0,0,0.05)",
              }}>
                <div style={{ padding: "12px 16px", borderBottom: `1px solid ${t.border}`, fontWeight: 700, fontSize: 13, color: t.textSub }}>
                  📸 Photo Tips
                </div>
                {[
                  { e: "📐", tip: "Shoot perpendicular to the wall for best results" },
                  { e: "🎯", tip: "Frame tight around the route section" },
                  { e: "💡", tip: "Avoid shooting into backlighting" },
                  { e: "🏷️", tip: "If routes overlap, zoom into one hold cluster" },
                ].map((item, i, arr) => (
                  <div key={i} style={{
                    display: "flex", gap: 12, padding: "11px 16px",
                    borderBottom: i < arr.length - 1 ? `1px solid ${t.border}` : "none",
                  }}>
                    <span style={{ fontSize: 16 }}>{item.e}</span>
                    <span style={{ fontSize: 13, color: t.textSub, lineHeight: 1.5 }}>{item.tip}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ══════════════ STATS ══════════════ */}
          {tab === "stats" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Top stats: Total beta + Contributors */}
              <div className="su" style={{
                background: t.bgCard, borderRadius: 16,
                border: `1.5px solid ${t.border}`, padding: "20px",
                boxShadow: "0 1px 8px rgba(0,0,0,0.05)",
                display: "flex", gap: 12,
              }}>
                {[
                  { val: stats?.total ?? "—", lbl: "Total beta", c: t.orange },
                  { val: stats?.contributors ?? "—", lbl: "Contributors", c: t.green },
                ].map(s => (
                  <div key={s.lbl} style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ fontWeight: 900, fontSize: 28, color: s.c, lineHeight: 1 }}>{s.val}</div>
                    <div style={{ fontSize: 11, color: t.textMuted, marginTop: 3, fontWeight: 600, letterSpacing: "0.5px" }}>
                      {s.lbl}
                    </div>
                  </div>
                ))}
              </div>

              {/* By Gym breakdown (with unknown/unclassified) */}
              <div className="su1" style={{
                background: t.bgCard, borderRadius: 16,
                border: `1.5px solid ${t.border}`, overflow: "hidden",
                boxShadow: "0 1px 8px rgba(0,0,0,0.05)",
              }}>
                <div style={{ padding: "12px 16px", borderBottom: `1px solid ${t.border}`, fontWeight: 700, fontSize: 13, color: t.textSub }}>
                  By Gym
                </div>
                {[
                  { key: "alpine", cnt: stats?.gym_alpine ?? 0 },
                  { key: "mainwall", cnt: stats?.gym_mainwall ?? 0 },
                  { key: "progression", cnt: stats?.gym_progression ?? 0 },
                  { key: "unknown", cnt: stats?.gym_unknown ?? 0 },
                ].map((item, i, arr) => {
                  const g = item.key === "unknown"
                    ? { full: "Unclassified", gc: t => t.textMuted }
                    : gymCfg(item.key);
                  const color = g.gc(t);
                  const total = stats?.total || 1;
                  const pct = Math.round((item.cnt / total) * 100);
                  return (
                    <div key={item.key} style={{
                      padding: "12px 16px",
                      borderBottom: i < arr.length - 1 ? `1px solid ${t.border}` : "none",
                    }}>
                      <div style={{
                        display: "flex", justifyContent: "space-between",
                        alignItems: "center", marginBottom: 6,
                      }}>
                        <span style={{ fontWeight: 700, fontSize: 13, color }}>{g.full}</span>
                        <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: t.text }}>
                          {item.cnt}
                        </span>
                      </div>
                      <div style={{ height: 6, background: t.bgSubtle, borderRadius: 3, overflow: "hidden" }}>
                        <div style={{
                          width: `${pct}%`, height: "100%", background: color,
                          borderRadius: 3, transition: "width 1s cubic-bezier(.4,0,.2,1)",
                        }} />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* By Source breakdown */}
              <div className="su2" style={{
                background: t.bgCard, borderRadius: 16,
                border: `1.5px solid ${t.border}`, padding: "16px",
                boxShadow: "0 1px 8px rgba(0,0,0,0.05)",
              }}>
                <div style={{ fontWeight: 700, fontSize: 13, color: t.textSub, marginBottom: 12 }}>
                  By Source
                </div>
                {(() => {
                  const total = stats?.total || 1;
                  const sources = [
                    { key: "tagged", cnt: stats?.source_tagged ?? 0, lbl: "Tagged", c: t.teal, e: "🏷️" },
                    { key: "official", cnt: stats?.source_official ?? 0, lbl: "Official", c: t.amber, e: "🏟️" },
                    { key: "contributor", cnt: stats?.source_contributor ?? 0, lbl: "Contributor", c: t.green, e: "👤" },
                  ];
                  return (
                    <div style={{ display: "flex", gap: 8 }}>
                      {sources.map(s => {
                        const pct = total > 0 ? Math.round((s.cnt / total) * 100) : 0;
                        return (
                          <div key={s.key} style={{
                            flex: s.cnt, minWidth: "60px", background: s.c + "18",
                            border: `1.5px solid ${s.c}33`,
                            borderRadius: 10, padding: "10px 8px", textAlign: "center",
                          }}>
                            <div style={{ fontSize: 12 }}>{s.e}</div>
                            <div style={{ fontWeight: 900, fontSize: 16, color: s.c, lineHeight: 1, marginTop: 2 }}>
                              {s.cnt}
                            </div>
                            <div style={{ fontSize: 9, color: s.c, fontWeight: 700, marginTop: 2, letterSpacing: "0.5px" }}>
                              {pct}%
                            </div>
                            <div style={{ fontSize: 9, color: s.c + "99", marginTop: 1 }}>
                              {s.lbl}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  );
                })()}
              </div>
            </div>
          )}

          {/* ══════════════ ABOUT ══════════════ */}
          {tab === "about" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div className="su" style={{
                background: t.bgCard, borderRadius: 16,
                border: `1.5px solid ${t.border}`, padding: "20px",
                boxShadow: "0 1px 8px rgba(0,0,0,0.05)",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
                  <div style={{
                    width: 44, height: 44, borderRadius: 12,
                    background: `linear-gradient(135deg,${t.orange},${t.amber})`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 22, boxShadow: `0 4px 16px ${t.orange}44`,
                  }}>🧗</div>
                  <div>
                    <div style={{ fontWeight: 900, fontSize: 18 }}>
                      Beta<span style={{ color: t.orange }}>Finder</span> CNX
                    </div>
                    <div style={{ fontSize: 12, color: t.textMuted, marginTop: 1 }}>
                      Chiang Mai Climbing · Thailand 🇹🇭
                    </div>
                  </div>
                </div>
                <p style={{ fontSize: 13, color: t.textSub, lineHeight: 1.7, marginBottom: 10 }}>
                  Snap a photo of a climbing wall → get ranked Instagram beta posts back instantly. Compare your project with climbs posted by the Chiang Mai gym community.
                </p>
                <p style={{ fontSize: 13, color: t.textSub, lineHeight: 1.7, marginBottom: 10 }}>
                  Made with 🤍 and lots of 🧗‍♂️ (and chalk dust) by the Chiang Mai climbing community.
                </p>
                <p style={{ fontSize: 13, color: t.textSub, lineHeight: 1.7 }}>
                  Inspired by the original{" "}
                  <span style={{ fontWeight: 700, color: t.orange }}>BetaScan</span>{" "}
                  by <span style={{ fontWeight: 700, color: t.teal }}>@thangman22</span>.
                </p>
              </div>

              {/* Tech Stack (hidden for now — uncomment to show) */}
              {/* <div className="su1" style={{
                background: t.bgCard, borderRadius: 16,
                border: `1.5px solid ${t.border}`, overflow: "hidden",
                boxShadow: "0 1px 8px rgba(0,0,0,0.05)",
              }}>
                <div style={{ padding: "12px 16px", borderBottom: `1px solid ${t.border}`, fontWeight: 700, fontSize: 13, color: t.textSub }}>
                  Tech Stack
                </div>
                {[
                  ["Embeddings", "CLIP ViT-B/32", t.teal],
                  ["Search", "FAISS cosine similarity", t.green],
                  ["Scraping", "instaloader · tagged feed", t.amber],
                  ["Video", "OpenCV keyframe extraction", t.orange],
                  ["Infra", "UM890 Pro + RTX 3060 eGPU", t.blue],
                ].map(([k, v, c], i, arr) => (
                  <div key={k} style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "10px 16px",
                    borderBottom: i < arr.length - 1 ? `1px solid ${t.border}` : "none",
                  }}>
                    <span style={{ fontSize: 12, color: t.textMuted, fontWeight: 600 }}>{k}</span>
                    <span style={{ fontSize: 12, color: c, fontWeight: 600 }}>{v}</span>
                  </div>
                ))}
              </div> */

              <div className="su1" style={{ display: "flex", gap: 10 }}>
                {[
                  { lbl: "GitHub", href: "https://github.com/spatipan/beta-finder", c: t.textSub },
                  { lbl: "@patipan_poty", href: "https://instagram.com/patipan_poty", c: t.orange },
                  { lbl: "@climb.with.poom", href: "https://instagram.com/climb.with.poom", c: t.amber },
                ].map(link => (
                  <a key={link.lbl} href={link.href} target="_blank" rel="noopener noreferrer"
                    style={{
                      flex: 1, textAlign: "center", padding: "12px 10px",
                      borderRadius: 10, border: `1.5px solid ${t.border}`,
                      background: t.bgSubtle, color: link.c,
                      fontSize: 11, fontWeight: 700, textDecoration: "none",
                      transition: "border-color .15s, background .15s",
                    }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = link.c; e.currentTarget.style.background = link.c + "08"; }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = t.border; e.currentTarget.style.background = t.bgSubtle; }}
                  >{link.lbl}</a>
                ))}
              </div>
            </div>
          )}

        </div>
      </div >
    </>
  );
}