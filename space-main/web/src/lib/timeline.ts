// Turns the API's timeline (readings already split into runs at Phase 3 data gaps) into chart rows. Nothing is
// computed: every row is a reading as returned, except one null row between runs so the line breaks at a gap
// instead of being drawn across it.
import type { TimelineRun } from "./api/types";

/** Log time on a UTC axis: a naive log timestamp is placed as written (never shifted into the browser's zone). */
export function logTimeMs(iso: string): number {
  return /(Z|[+-]\d{2}:\d{2})$/.test(iso) ? Date.parse(iso) : Date.parse(`${iso.slice(0, 23)}Z`);
}

export function axisTime(ms: number): string {
  return new Date(ms).toISOString().slice(5, 16).replace("T", " ");
}

export type Row = { t: number; temp: number | null; ambient: number | null; line: number | null };

export function toRows(runs: TimelineRun[]): Row[] {
  const rows: Row[] = [];
  runs.forEach((run, i) => {
    if (i > 0) rows.push({ t: logTimeMs(runs[i - 1].end) + 1, temp: null, ambient: null, line: null }); // the gap
    for (const p of run.points) rows.push({ t: logTimeMs(p.ts), temp: p.temp_c, ambient: p.ambient_c, line: p.source_line });
  });
  return rows;
}
