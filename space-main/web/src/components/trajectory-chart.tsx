import type { TrajectoryPoint } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";

// A plain SVG line of the PREDICTED trajectory. It is rendered only from a PredictionSuccess, whose type is the
// only one with a trajectory; a refusal has no such field.
export function TrajectoryChart({
  points,
  thresholdC,
  crossingMinutes,
}: {
  points: TrajectoryPoint[];
  thresholdC: number | null;
  crossingMinutes: number | null;
}) {
  const W = 640;
  const H = 260;
  const pad = { l: 48, r: 16, t: 16, b: 36 };
  const xs = points.map((p) => p.minutes);
  const ys = points.map((p) => p.temp_c).concat(thresholdC === null ? [] : [thresholdC]);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  let y0 = Math.min(...ys);
  let y1 = Math.max(...ys);
  if (y1 - y0 < 1) {
    y0 -= 0.5;
    y1 += 0.5;
  }
  const sx = (m: number) => pad.l + ((m - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r);
  const sy = (t: number) => H - pad.b - ((t - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const line = points.map((p) => `${sx(p.minutes).toFixed(1)},${sy(p.temp_c).toFixed(1)}`).join(" ");
  return (
    <figure className="rounded-lg border border-slate-200 bg-white p-4" data-testid="trajectory-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Predicted room-air temperature over ${formatNumber(x1, 0)} minutes`} className="h-auto w-full">
        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="#94a3b8" />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="#94a3b8" />
        <text x={pad.l - 6} y={sy(y1) + 4} textAnchor="end" fontSize="11" fill="#475569">{formatNumber(y1, 1)}</text>
        <text x={pad.l - 6} y={sy(y0) + 4} textAnchor="end" fontSize="11" fill="#475569">{formatNumber(y0, 1)}</text>
        <text x={pad.l} y={H - 12} fontSize="11" fill="#475569">+{formatNumber(x0, 0)} min</text>
        <text x={W - pad.r} y={H - 12} textAnchor="end" fontSize="11" fill="#475569">+{formatNumber(x1, 0)} min</text>
        {thresholdC !== null && (
          <g>
            <line x1={pad.l} x2={W - pad.r} y1={sy(thresholdC)} y2={sy(thresholdC)} stroke="#be123c" strokeDasharray="6 4" />
            <text x={W - pad.r} y={sy(thresholdC) - 4} textAnchor="end" fontSize="11" fill="#be123c">threshold {formatNumber(thresholdC, 1)} °C</text>
          </g>
        )}
        <polyline points={line} fill="none" stroke="#0369a1" strokeWidth="2" />
        {crossingMinutes !== null && thresholdC !== null && (
          <circle cx={sx(crossingMinutes)} cy={sy(thresholdC)} r="4" fill="#be123c" />
        )}
      </svg>
      <figcaption className="mt-2 text-xs text-slate-500">
        °C, minutes from the prediction start. Every point is PREDICTED (a model output), not a measurement.
      </figcaption>
    </figure>
  );
}
