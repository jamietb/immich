export type SimilarSession = { baseId: string; ids: string[]; index: number; meta?: Record<string, unknown> };

let current: SimilarSession | null = null;

export function createSimilarSession(s: SimilarSession) { current = { ...s }; }
export function clearSimilarSession() { current = null; }
export function hasSimilarSession() { return !!current; }
export function getSimilarSession(): SimilarSession | null { return current ? { ...current } : null; }
export function setIndex(i: number) { if (!current) return; current.index = Math.max(0, Math.min(current.ids.length - 1, i)); }
export function step(delta: number) { if (!current) return null; setIndex(current.index + delta); return current!.ids[current!.index]; }
