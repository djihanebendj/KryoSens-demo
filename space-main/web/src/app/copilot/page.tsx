import { CopilotStatus } from "./copilot-status";
import { FREE_TEXT_NOTICE, GuidedStart, SAFETY_NOTICE } from "./guided-start";

export const metadata = { title: "Technician Copilot — KryoSens" };

export default function Page() {
  return (
    <div className="space-y-6">
      <p role="note" data-testid="free-text-notice" className="rounded-lg border border-slate-300 bg-slate-50 p-4 text-sm text-slate-800">
        {FREE_TEXT_NOTICE}
      </p>
      <p data-testid="experiment-note" className="text-sm text-slate-600">
        An experimental copilot prototype is available separately in experiments/copilot-kyrosync (see README).
      </p>
      <CopilotStatus />
      <GuidedStart />
      <p role="note" data-testid="safety-notice" className="rounded-lg border border-rose-300 bg-rose-50 p-4 text-sm text-rose-900">
        {SAFETY_NOTICE}
      </p>
    </div>
  );
}
