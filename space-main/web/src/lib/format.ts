// Display helpers. Log timestamps are shown exactly as the backend reports them: a naive timestamp is the
// log's own local time of unknown zone (Phase 2 time basis) and must NOT be shifted into the browser's zone.

const AWARE = /(Z|[+-]\d{2}:\d{2})$/;

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "—";
  if (AWARE.test(iso)) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return `${d.toISOString().slice(0, 19).replace("T", " ")} UTC`;
  }
  return iso.slice(0, 19).replace("T", " ");
}

export function formatNumber(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

export function formatTemp(v: number | null | undefined, digits = 2): string {
  const s = formatNumber(v, digits);
  return s === "—" ? s : `${s} °C`;
}

export function formatMinutes(v: number | null | undefined): string {
  const s = formatNumber(v, 1);
  return s === "—" ? s : `${s} min`;
}

export function shortHash(h: string): string {
  return h.length > 16 ? `${h.slice(0, 16)}…` : h;
}

export function humanize(code: string): string {
  return code.toLowerCase().replace(/_/g, " ");
}
