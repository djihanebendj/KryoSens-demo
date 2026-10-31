"use client";

import { CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { TimelineView } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";
import { axisTime, logTimeMs, toRows } from "@/lib/timeline";

/** Measured temperature (and ambient, when recorded) with Phase 3 data gaps and threshold excursions marked.
 * Readings are drawn as measured: no smoothing, no interpolation, and the line is broken at every data gap. */
export function TimelineChart({ view, width }: { view: TimelineView; width?: number }) {
  const rows = toRows(view.runs);
  const excursions = view.events.filter((e) => e.type === "THRESHOLD_EXCURSION");
  const chart = (
    <LineChart data={rows} width={width} height={320} margin={{ top: 12, right: 24, bottom: 8, left: 8 }}>
      <CartesianGrid stroke="#e2e8f0" />
      <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={axisTime} fontSize={11} />
      <YAxis unit=" °C" fontSize={11} width={64} domain={["auto", "auto"]} />
      <Tooltip
        labelFormatter={(t) => `${axisTime(Number(t))} (log time)`}
        formatter={(v, name) => [v === null || v === undefined ? "no reading" : `${formatNumber(Number(v), 2)} °C`, name]}
      />
      <Legend />
      {view.gaps.map((g) => (
        <ReferenceArea key={g.event_id} x1={logTimeMs(g.start)} x2={logTimeMs(g.end)} fill="#94a3b8" fillOpacity={0.25} label={{ value: "no data", fontSize: 10, fill: "#475569" }} />
      ))}
      {excursions.map((e) => (
        <ReferenceArea key={e.event_id} x1={logTimeMs(e.start)} x2={logTimeMs(e.end)} fill="#f59e0b" fillOpacity={0.12} />
      ))}
      {view.upper_threshold_c !== null && <ReferenceLine y={view.upper_threshold_c} stroke="#be123c" strokeDasharray="6 4" label={{ value: `upper limit ${view.upper_threshold_c} °C`, fontSize: 10, fill: "#be123c", position: "insideTopRight" }} />}
      {view.lower_threshold_c !== null && <ReferenceLine y={view.lower_threshold_c} stroke="#1d4ed8" strokeDasharray="6 4" label={{ value: `lower limit ${view.lower_threshold_c} °C`, fontSize: 10, fill: "#1d4ed8", position: "insideBottomRight" }} />}
      <Line name="room air (measured)" type="linear" dataKey="temp" stroke="#0369a1" dot={false} strokeWidth={1.5} connectNulls={false} isAnimationActive={false} />
      {view.ambient_available && <Line name="ambient (measured)" type="linear" dataKey="ambient" stroke="#9333ea" dot={false} strokeWidth={1} connectNulls={false} isAnimationActive={false} />}
    </LineChart>
  );
  return (
    <div data-testid="timeline-chart" className="h-[320px] w-full">
      {width ? chart : <ResponsiveContainer width="100%" height={320}>{chart}</ResponsiveContainer>}
    </div>
  );
}
