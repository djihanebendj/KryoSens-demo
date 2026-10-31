"use client";

// Landing page. Layout and visual direction come from design/figma-landing-1.html (Tinhinane's mockup);
// the copy is rewritten so every sentence matches what the build actually does, and every button goes to
// a route that exists. The mockup's CDN assets (Tailwind, Chart.js, Lucide, Google Fonts) are not used:
// styling is a local CSS module, so the page works offline and ships nothing from a third party.
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { ApiFailure } from "@/lib/api/client";
import { useCase } from "@/lib/case-context";

import styles from "./landing.module.css";

const SAMPLE = "SIMULATED_power_cut_22h_1min.csv";

export default function Landing() {
  const { api, setStatus } = useCase();
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function tryDemo() {
    setBusy(true);
    setError(null);
    try {
      const samples = await api.samples();
      const pick = samples.find((s) => s.filename === SAMPLE) ?? samples[0];
      if (!pick) throw new Error("the service returned no bundled sample");
      setStatus(await api.createSampleCase(pick.sample_id));
      router.push("/quality");
    } catch (e) {
      setError(
        e instanceof ApiFailure
          ? `Could not start the demo: ${e.message}. Is the API running?`
          : `Could not start the demo: ${(e as Error).message}`,
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.inner}>
        <header className={styles.bar}>
          <div className={styles.brand}>
            Kryo<span>Sens</span>
          </div>
          <span className={styles.tag}>evidence-first cold-room analysis</span>
          <nav className={styles.barLinks} aria-label="Landing">
            <Link href="/dashboard">Dashboard</Link>
            <Link href="/upload">Upload</Link>
            <Link href="/next-check">Next best check</Link>
            <Link href="/copilot">Copilot</Link>
          </nav>
        </header>

        <section className={styles.hero}>
          <p className={styles.kicker}>Cold-room temperature logs</p>
          <h1 className={styles.h1}>
            Read a cold-room log, and say only what the data supports.
          </h1>
          <p className={styles.lead}>
            Upload a temperature log. KryoSens checks whether the file can be analysed at all, splits it
            into events, forecasts how the room warms with no active cooling or refuses and says why,
            then lists candidate explanations and the single check worth doing next. Nothing is reported
            as a confirmed fault until a technician confirms it.
          </p>
          <div className={styles.ctaRow}>
            <button type="button" className={styles.primary} onClick={() => void tryDemo()} disabled={busy}>
              {busy ? "Starting the demo…" : "Try the demo (simulated log)"}
            </button>
            <Link href="/upload" className={styles.secondary}>
              Upload my own log
            </Link>
            <Link href="/dashboard" className={styles.secondary}>
              Open the dashboard
            </Link>
          </div>
          {error && (
            <p className={styles.err} role="alert">
              {error}
            </p>
          )}
        </section>

        <div className={styles.grid}>
          <article className={styles.card}>
            <h3>Data quality first</h3>
            <p>
              Gaps, duplicates, out-of-order rows, unknown time zones and flatlines are reported before
              anything is analysed. A log that cannot carry the question is refused.
            </p>
            <Link href="/quality">Data quality →</Link>
          </article>
          <article className={styles.card}>
            <h3>Physics, or a refusal</h3>
            <p>
              A first-order thermal model is fitted to periods with no active cooling and must beat two
              simple baselines on held-out data. When it cannot, the answer is a refusal with its reason,
              not a number.
            </p>
            <Link href="/prediction">Prediction →</Link>
          </article>
          <article className={styles.card}>
            <h3>Next best check</h3>
            <p>
              Candidate explanations are assessed only against recorded evidence, then the checks are
              ranked by how many open items each resolves, the effort it costs and the access it needs.
            </p>
            <Link href="/next-check">Next best check →</Link>
          </article>
        </div>

        <section className={styles.section}>
          <h2>How a case runs</h2>
          <p>Ten steps, each reachable from the navigation once the step before it has data.</p>
          <ol className={styles.steps}>
            <li><span className={styles.num}>01</span><Link href="/upload">Upload a log, or load a bundled simulated sample</Link></li>
            <li><span className={styles.num}>02</span><Link href="/quality">Read the data-quality verdict</Link></li>
            <li><span className={styles.num}>03</span><Link href="/timeline">Review the event timeline</Link></li>
            <li><span className={styles.num}>04</span><Link href="/prediction">Run the no-cooling forecast, or read its refusal</Link></li>
            <li><span className={styles.num}>05</span><Link href="/evidence">Separate facts from inferences on the evidence board</Link></li>
            <li><span className={styles.num}>06</span><Link href="/next-check">Take the next best check</Link></li>
            <li><span className={styles.num}>07</span><Link href="/investigation">Work the hypotheses in the investigation view</Link></li>
            <li><span className={styles.num}>08</span><Link href="/incident">Confirm a finding as a technician</Link></li>
            <li><span className={styles.num}>09</span><Link href="/reports">Compose the report from confirmed findings only</Link></li>
            <li><span className={styles.num}>10</span><Link href="/copilot">Use the rule-based Guided Start (free-text copilot disabled)</Link></li>
          </ol>
        </section>

        <section className={styles.section}>
          <h2>Technician Copilot</h2>
          <p>
            The free-text copilot is <strong>disabled in this build</strong>: the server answers
            FEATURE_DISABLED and no page can switch it on. An experimental Gemini-based prototype is kept
            separately in experiments/copilot-kyrosync and is not part of this app. <strong>Guided Start</strong>
            is always available instead: it is rule-based, with no language model, and shows the backend&apos;s
            own next check and proposed hypotheses.
          </p>
          <div className={styles.ctaRow}>
            <Link href="/copilot" className={styles.secondary}>
              Open the Copilot page
            </Link>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.limits}>
            <h2>What this build does not do</h2>
            <ul>
              <li>No validation on a real cold room, and no claim of real forecast accuracy.</li>
              <li>From the web app, the only equipment signal read is the <code>compressor_state</code> column; door, defrost and power records are a command-line input only.</li>
              <li>Hypotheses are never confirmed faults; a technician confirms a finding before it reaches a report.</li>
              <li>Uploaded data is always treated as unvalidated, whatever the uploader declares.</li>
              <li>Case data lives in server memory and is deleted after two hours.</li>
              <li>No Arabic support: the interface and the copilot are English only.</li>
            </ul>
          </div>
        </section>

        <footer className={styles.foot}>
          KryoSens · evidence-first cold-room decision support · landing design by Tinhinane Aichoune
        </footer>
      </div>
    </div>
  );
}
