export type Point = [number, number]
export interface SearchResult { id: number; label: string; dataset: 'Pets' | 'Flowers'; similarity: number; thumbnail_url: string; point: Point }
export interface SearchResponse { results: SearchResult[]; query_point: Point; timings: { embedding_ms: number; ranking_ms: number; total_ms: number }; index_id: string }
export interface Health { status: string; dataset_size: number; classes: number; index_id: string; model: string }
export const backend = (import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '')
const key = import.meta.env.VITE_API_KEY || ''
export const maxUploadMB = Number(import.meta.env.VITE_MAX_UPLOAD_MB || 10)

async function checked(response: Response) {
  if (response.ok) return response
  const body = await response.json().catch(() => null)
  const detail = typeof body?.detail === 'string' ? body.detail : 'The request could not be completed.'
  const message = response.status === 429 ? `${detail} Try again in ${response.headers.get('Retry-After') || '60'} seconds.` : detail
  throw new Error(message)
}
function authorizedHeaders() {
  if (!key || key === 'replace-with-a-random-local-key') throw new Error('Set VITE_API_KEY in frontend/.env to match the backend key, then restart the frontend.')
  return { 'X-API-Key': key }
}
export async function health(signal: AbortSignal): Promise<Health> {
  return (await checked(await fetch(`${backend}/health`, { signal }))).json()
}
export async function imageSearch(file: File, k: number, signal: AbortSignal): Promise<SearchResponse> {
  const data = new FormData(); data.append('file', file)
  return (await checked(await fetch(`${backend}/search?top_k=${k}`, { method: 'POST', headers: authorizedHeaders(), body: data, signal }))).json()
}
export async function textSearch(query: string, k: number, signal: AbortSignal): Promise<SearchResponse> {
  return (await checked(await fetch(`${backend}/search/text`, { method: 'POST', headers: { ...authorizedHeaders(), 'Content-Type': 'application/json' }, body: JSON.stringify({ query, top_k: k }), signal }))).json()
}
export async function thumbnail(path: string, signal: AbortSignal): Promise<Blob> {
  return (await checked(await fetch(`${backend}${path}`, { headers: authorizedHeaders(), signal }))).blob()
}
