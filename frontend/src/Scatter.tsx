import { useState } from 'react'
import type { SearchResponse, SearchResult } from './api'

export function Scatter({ data, active, setActive }: { data: SearchResponse; active: number | null; setActive: (id: number | null) => void }) {
  const [hovered, setHovered] = useState<SearchResult | null>(null)
  const points = [...data.results.map(r => r.point), data.query_point]
  const xs = points.map(p => p[0]), ys = points.map(p => p[1])
  const cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), .05) * 1.4
  const x = (n: number) => 200 + (n - cx) / span * 290
  const y = (n: number) => 145 - (n - cy) / span * 220
  return <div className="scatter-content">
    <svg viewBox="0 0 400 290" role="img" aria-label="Two dimensional PCA projection of the query and nearest neighbors">
      <defs><pattern id="dots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r="1" fill="#dfe2ed"/></pattern></defs>
      <rect x="16" y="10" width="368" height="260" rx="12" fill="url(#dots)"/>
      <line x1="40" y1="245" x2="367" y2="245" stroke="#e5e7ef"/><line x1="40" y1="25" x2="40" y2="245" stroke="#e5e7ef"/>
      {data.results.map(r => <g key={`line-${r.id}`} opacity={active === r.id ? .7 : .13}><line x1={x(data.query_point[0])} y1={y(data.query_point[1])} x2={x(r.point[0])} y2={y(r.point[1])} stroke="#7c5be5" strokeDasharray="3 5"/></g>)}
      {data.results.map((r, i) => <g key={r.id} tabIndex={0} role="button" aria-label={`Neighbor ${i + 1}: ${r.label}`} onMouseEnter={() => { setHovered(r); setActive(r.id) }} onMouseLeave={() => { setHovered(null); setActive(null) }} onFocus={() => { setHovered(r); setActive(r.id) }} onBlur={() => { setHovered(null); setActive(null) }} onClick={() => setActive(r.id)}>
        <circle cx={x(r.point[0])} cy={y(r.point[1])} r={active === r.id ? 10 : 7} fill={r.dataset === 'Pets' ? '#9682ea' : '#57bba7'} stroke="white" strokeWidth="2"/>
        <title>{i + 1}. {r.label} · {(r.similarity * 100).toFixed(1)}% cosine similarity</title>
      </g>)}
      <circle cx={x(data.query_point[0])} cy={y(data.query_point[1])} r="16" fill="#7554e8" opacity=".13"/>
      <path d={`M ${x(data.query_point[0])} ${y(data.query_point[1]) - 8} l 8 8 l -8 8 l -8 -8 z`} fill="#6441d6" stroke="white" strokeWidth="2"/>
      <text x="340" y="280" fontSize="10" fill="#959bad">PC 1 →</text><text x="12" y="25" fontSize="10" fill="#959bad">PC 2</text>
    </svg>
    <div className="plot-legend"><span><i className="diamond"/>Your query</span><span><i className="pet-dot"/>Pets</span><span><i className="flower-dot"/>Flowers</span></div>
    <p className="plot-caption">{hovered ? `${hovered.label} · ${(hovered.similarity * 100).toFixed(1)}% similarity` : 'Hover a point to explore a match. PCA is a 2D approximation of the 512D search space.'}</p>
  </div>
}
