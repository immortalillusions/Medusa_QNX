"use client";

import { useEffect, useState } from "react";

type Status = {
  active: boolean;
  reason: string;
  face_count?: number | null;
  snooper_count?: number;
  mode?: string;
  laptop_active?: boolean | null;
  laptop_age_s?: number | null;
  aim_target?: [number, number] | null;
  fps?: number | null;
  pan_state?: string;
  learned_points?: number | null;
};

// Served by the high-priority control process on the Pi at :8082/control.json.
// Every field is optional; the panel renders whatever is present.
type Control = {
  state?: "SAFE" | "ACTIVE";
  loop_hz?: number;
  period_ms_mean?: number;
  jitter_ms_p99?: number;
  period_ms_max?: number;
  missed_deadlines?: number;
  target_age_s?: number;
  ai_alive?: boolean;
  arduino?: { link_ok?: boolean; armed?: boolean; pan?: number; tilt?: number; age_s?: number };
  safe_reason?: string;
  stress?: { running?: boolean; seconds_left?: number };
};

type Source = ReturnType<typeof useSource>;
type Light = "ok" | "warn" | "bad" | "unknown";

const CONTROL_PORT = 8082;
const SPARK_SAMPLES = 60;

function remember(key: string, value?: string) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {}
  return null;
}

const isNum = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);
const fmt = (n: unknown, digits: number, unit = "") => (isNum(n) ? `${n.toFixed(digits)}${unit}` : "—");

// Polls `<url><path>` as JSON, waiting `intervalMs` between polls and aborting after `timeoutMs`.
function usePoll<T>(url: string, path: string, intervalMs: number, timeoutMs: number) {
  const [data, setData] = useState<T | null>(null);
  const [connected, setConnected] = useState(false);
  const [tried, setTried] = useState(false);

  useEffect(() => {
    if (!url) {
      setData(null);
      setConnected(false);
      return;
    }
    let cancelled = false;
    let request: AbortController | undefined;
    let timer: number;
    const poll = async () => {
      request = new AbortController();
      const timeout = window.setTimeout(() => request?.abort(), timeoutMs);
      try {
        const response = await fetch(url + path, { cache: "no-store", signal: request.signal });
        if (!response.ok) throw new Error();
        const next: T = await response.json();
        if (cancelled) return;
        setData(next);
        setConnected(true);
      } catch {
        if (cancelled) return;
        setData(null);
        setConnected(false);
      } finally {
        window.clearTimeout(timeout);
        if (!cancelled) setTried(true);
      }
      if (!cancelled) timer = window.setTimeout(poll, intervalMs);
    };
    void poll();
    return () => {
      cancelled = true;
      request?.abort();
      window.clearTimeout(timer);
    };
  }, [url, path, intervalMs, timeoutMs]);

  return { data, connected, tried };
}

function useSource(storageKey: string, defaultUrl: string) {
  const [input, setInput] = useState(defaultUrl);
  const [url, setUrl] = useState(defaultUrl);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const poll = usePoll<Status>(url, "/status.json", 400, 2000);
  const connected = poll.connected;
  const status: Status = connected && poll.data
    ? poll.data
    : { active: false, reason: poll.tried ? "source unavailable" : "waiting" };

  useEffect(() => {
    const saved = remember(storageKey) || defaultUrl;
    setInput(saved);
    setUrl(saved);
  }, [storageKey, defaultUrl]);

  useEffect(() => {
    if (!connected) return;
    setPreviewFailed(false);
    setAttempt((a) => a + 1);
  }, [connected]);

  useEffect(() => {
    if (!previewFailed) return;
    const timer = window.setTimeout(() => {
      setPreviewFailed(false);
      setAttempt((a) => a + 1);
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [previewFailed]);

  const connect = () => {
    const normalized = input.trim().replace(/\/$/, "");
    remember(storageKey, normalized);
    setUrl(normalized);
    setPreviewFailed(false);
    setAttempt((a) => a + 1);
  };

  return { input, setInput, url, status, connected, previewFailed, setPreviewFailed, attempt, connect };
}

// Same host as the turret, port 8082.
function controlUrlFrom(turretUrl: string) {
  try {
    const parsed = new URL(turretUrl);
    return `${parsed.protocol}//${parsed.hostname}:${CONTROL_PORT}`;
  } catch {
    return "";
  }
}

// Control base URL: derived from the turret address unless overridden (override remembered in localStorage).
function useControlUrl(turretUrl: string) {
  const storageKey = "medusa-control-url";
  const [override, setOverride] = useState("");
  const [input, setInput] = useState("");

  useEffect(() => {
    const saved = remember(storageKey) || "";
    setOverride(saved);
    setInput(saved);
  }, []);

  const derived = controlUrlFrom(turretUrl);
  const save = () => {
    const normalized = input.trim().replace(/\/$/, "");
    remember(storageKey, normalized);
    setOverride(normalized);
    setInput(normalized);
  };
  const reset = () => {
    remember(storageKey, "");
    setOverride("");
    setInput("");
  };

  return { url: override || derived, derived, override, input, setInput, save, reset };
}

function details(status: Status) {
  const lines: string[] = [];
  if (typeof status.face_count === "number") {
    lines.push(`${status.face_count} ${status.face_count === 1 ? "face" : "faces"} detected`);
  }
  if (typeof status.snooper_count === "number") {
    lines.push(`${status.snooper_count} ${status.snooper_count === 1 ? "snooper" : "snoopers"}`);
  }
  if (status.mode) lines.push(`Mode: ${status.mode}`);
  if (isNum(status.fps)) lines.push(`Frame rate: ${status.fps.toFixed(1)} fps`);
  if (status.laptop_active !== undefined) {
    const age = typeof status.laptop_age_s === "number" ? ` (${status.laptop_age_s.toFixed(1)} s ago)` : "";
    const said = status.laptop_active === null ? "no data" : status.laptop_active ? "peeper" : "clear";
    lines.push(`Laptop says: ${said}${age}`);
  }
  if (status.aim_target !== undefined) {
    const target = status.aim_target;
    lines.push(target ? `Aiming at x ${target[0].toFixed(2)}, y ${target[1].toFixed(2)}` : "No target");
  }
  if (status.pan_state === "fallback") lines.push("Aim: FALLBACK (laptop)");
  if (isNum(status.learned_points)) lines.push(`Learned map: ${status.learned_points} points`);
  return lines;
}

function LinkLight({ label, light, value }: { label: string; light: Light; value: React.ReactNode }) {
  return <div className="light">
    <span className={`dot ${light}`} aria-hidden />
    <div><p>{label}</p><small>{value}</small></div>
  </div>;
}

function Sparkline({ values, nominal }: { values: number[]; nominal?: number }) {
  const width = 240, height = 44;
  const top = Math.max(1, ...values, isNum(nominal) ? nominal * 1.5 : 0);
  const y = (v: number) => height - (v / top) * height;
  const offset = SPARK_SAMPLES - values.length;
  const points = values.map((v, i) => `${((offset + i) * width) / (SPARK_SAMPLES - 1)},${y(v).toFixed(1)}`).join(" ");
  return <svg className="spark" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img"
    aria-label="Recent control loop timing">
    {isNum(nominal) && <line x1={0} x2={width} y1={y(nominal)} y2={y(nominal)} className="spark-nominal" />}
    {values.length > 1 && <polyline points={points} className="spark-line" />}
  </svg>;
}

function SafetyPanel({ turret, control, controlUrl }: {
  turret: Source;
  control: ReturnType<typeof usePoll<Control>>;
  controlUrl: ReturnType<typeof useControlUrl>;
}) {
  const c = control.connected ? control.data ?? {} : null;
  const [history, setHistory] = useState<number[]>([]);
  const [stressPending, setStressPending] = useState(false);
  const [stressError, setStressError] = useState<string | null>(null);

  const sparkMetric = isNum(c?.period_ms_max) ? "period max" : "p99 jitter";
  useEffect(() => {
    const sample = control.data?.period_ms_max ?? control.data?.jitter_ms_p99;
    if (isNum(sample)) setHistory((h) => [...h.slice(-(SPARK_SAMPLES - 1)), sample]);
  }, [control.data]);

  // Big indicator
  const active = c?.state === "ACTIVE";
  const stateClass = !c || !c.state ? "alert safety-state" : active ? "alert safety-state active" : "alert safety-state safe";
  const stateText = !c ? "CONTROL OFFLINE" : c.state ?? "UNKNOWN";
  const stateReason = !c
    ? `No answer from ${controlUrl.url || "(no address)"}/control.json`
    : c.safe_reason ?? (active ? "armed — aim is live" : "");

  // Link lights
  const laptopAge = turret.connected ? turret.status.laptop_age_s : undefined;
  const laptopLight: Light = !isNum(laptopAge) ? "unknown" : laptopAge < 0.5 ? "ok" : "bad";
  const turretFps = turret.connected ? turret.status.fps : undefined;
  const turretLight: Light = !turret.connected ? "bad" : !isNum(turretFps) || turretFps > 0 ? "ok" : "warn";
  const aiLight: Light = c?.ai_alive === true ? "ok" : c?.ai_alive === false ? "bad" : "unknown";
  const arduino = c?.arduino;
  const arduinoLight: Light = arduino?.link_ok === true ? "ok" : arduino?.link_ok === false ? "bad" : "unknown";
  const pose = isNum(arduino?.pan) || isNum(arduino?.tilt) ? ` · pan ${fmt(arduino?.pan, 0)} tilt ${fmt(arduino?.tilt, 0)}` : "";

  // Stress test
  const stressRunning = c?.stress?.running === true;
  const runStress = async () => {
    setStressPending(true);
    setStressError(null);
    const request = new AbortController();
    const timeout = window.setTimeout(() => request.abort(), 3000);
    try {
      const response = await fetch(`${controlUrl.url}/stress?seconds=20`, { method: "POST", signal: request.signal });
      if (!response.ok) setStressError(`control loop refused (HTTP ${response.status})`);
    } catch {
      setStressError("control loop offline");
    } finally {
      window.clearTimeout(timeout);
      setStressPending(false);
    }
  };
  const stressNote = stressRunning ? `${fmt(c?.stress?.seconds_left, 0, " s")} left`
    : !c ? "control loop offline" : stressError;

  return <section className="safety">
    <div className="safety-top">
      <aside className={stateClass}>
        <p className="eyebrow">SAFETY</p>
        <strong>{stateText}</strong>
        {stateReason && <small>{stateReason}</small>}
      </aside>

      <div className="lights">
        <LinkLight label="Laptop link" light={laptopLight}
          value={isNum(laptopAge) ? `${laptopAge.toFixed(2)} s ago` : "no data"} />
        <LinkLight label="Turret camera" light={turretLight}
          value={!turret.connected ? "offline" : isNum(turretFps) ? `${turretFps.toFixed(1)} fps` : "connected"} />
        <LinkLight label="AI process" light={aiLight}
          value={!c ? "unknown" : c.ai_alive === undefined ? "no data"
            : `${c.ai_alive ? "alive" : "stalled"}${isNum(c.target_age_s) ? ` · target ${c.target_age_s.toFixed(2)} s` : ""}`} />
        <LinkLight label="Control loop" light={c ? "ok" : "bad"} value={c ? fmt(c.loop_hz, 1, " Hz") : "offline"} />
        <LinkLight label="Arduino" light={arduinoLight}
          value={!arduino ? (c ? "no data" : "unknown") : <>
            {arduino.armed !== undefined && <b className={arduino.armed ? "armed" : undefined}>
              {arduino.armed ? "ARMED" : "DISARMED"}</b>}{pose}
          </>} />
      </div>
    </div>

    <div className="realtime">
      <div className="rt-head">
        <p className="eyebrow">REAL-TIME</p>
        <small>AI slows down, control loop doesn&apos;t.</small>
      </div>
      <dl className="rt-stats">
        <div><dt>Period mean</dt><dd>{fmt(c?.period_ms_mean, 2, " ms")}</dd></div>
        <div><dt>p99 jitter</dt><dd>{fmt(c?.jitter_ms_p99, 2, " ms")}</dd></div>
        <div><dt>Period max</dt><dd>{fmt(c?.period_ms_max, 2, " ms")}</dd></div>
        <div><dt>Missed</dt><dd className={isNum(c?.missed_deadlines) && c.missed_deadlines > 0 ? "armed" : undefined}>
          {fmt(c?.missed_deadlines, 0)}</dd></div>
        <div><dt>AI fps</dt><dd>{fmt(turretFps, 1)}</dd></div>
      </dl>
      <div className="rt-spark">
        <Sparkline values={history} nominal={c?.period_ms_mean ?? (isNum(c?.loop_hz) && c.loop_hz > 0 ? 1000 / c.loop_hz : undefined)} />
        <small>{sparkMetric}, last {SPARK_SAMPLES} samples{history.length ? ` · now ${history[history.length - 1].toFixed(2)} ms` : ""}</small>
      </div>
      <div className="rt-stress">
        <button onClick={runStress} disabled={!c || stressRunning || stressPending}>
          {stressRunning ? "Stress running…" : stressPending ? "Starting…" : "CPU stress test (20 s)"}
        </button>
        {stressNote && <small>{stressNote}</small>}
      </div>
    </div>

    <details className="advanced">
      <summary>Advanced</summary>
      <div className="setup">
        <label htmlFor="control-url">Control loop address</label>
        <div>
          <input id="control-url" value={controlUrl.input} placeholder={controlUrl.derived || "http://<pi-host>:8082"}
            onChange={(e) => controlUrl.setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && controlUrl.save()} />
          <button onClick={controlUrl.save}>Save</button>
          {controlUrl.override && <button className="secondary" onClick={controlUrl.reset}>Reset</button>}
        </div>
        <small>{controlUrl.override ? "Override in use." : `Derived from the turret address (port ${CONTROL_PORT}).`} Polling {controlUrl.url || "nothing"}/control.json</small>
      </div>
    </details>
  </section>;
}

function SourcePanel({ id, title, hint, source }: { id: string; title: string; hint: string; source: Source }) {
  const { status, connected } = source;
  return <section className="source">
    <div className="source-head">
      <h2>{title}</h2>
      <span className={connected ? "chip connected" : "chip"}>{connected ? "Connected" : "Offline"}</span>
    </div>

    <div className="setup">
      <label htmlFor={id}>Address</label>
      <div>
        <input id={id} value={source.input} onChange={(e) => source.setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && source.connect()} />
        <button onClick={source.connect}>Connect</button>
      </div>
      <small>{hint}</small>
    </div>

    <div className="video">
      {source.previewFailed ? <p>Reconnecting camera preview…</p> :
        <img key={source.attempt} src={source.url + "/stream.mjpg"} alt={`${title} preview`}
          onError={() => source.setPreviewFailed(true)} />}
    </div>

    <aside className={connected && status.active ? "alert active" : "alert"}>
      <p className="eyebrow">CURRENT STATE</p>
      <strong>{connected ? (status.active ? "PEEPER" : "CLEAR") : "UNAVAILABLE"}</strong>
      {connected && details(status).map((line) => <p key={line.split(":")[0]}>{line}</p>)}
      <small>{status.reason}</small>
    </aside>
  </section>;
}

export default function Home() {
  const laptop = useSource("medusa-laptop-url", "http://localhost:8081");
  const turret = useSource("medusa-turret-url", "http://172.20.10.2:8080");
  const controlUrl = useControlUrl(turret.url);
  const control = usePoll<Control>(controlUrl.url, "/control.json", 250, 1000);

  const deciding = turret.connected ? turret : laptop.connected ? laptop : null;
  const peeper = deciding?.status.active ?? false;
  const headline = !deciding ? "NO SOURCE CONNECTED" : peeper ? "PEEPER — docs switched, turret aiming" : "CLEAR";
  const decidedBy = deciding === turret ? "turret (QNX)" : deciding === laptop ? "laptop webcam" : "neither source";

  return <main>
    <header>
      <div><p className="eyebrow">LOCAL QNX SETUP VIEW</p><h1>Medusa</h1></div>
    </header>

    <section className={peeper ? "alert banner active" : "alert banner"}>
      <p className="eyebrow">SYSTEM</p>
      <strong>{headline}</strong>
      <small>Decided by {decidedBy}</small>
    </section>

    <SafetyPanel turret={turret} control={control} controlUrl={controlUrl} />

    <section className="sources">
      <SourcePanel id="laptop-url" title="Laptop webcam — detection" source={laptop}
        hint="The Mac detector serves /stream.mjpg and /status.json." />
      <SourcePanel id="turret-url" title="Turret camera — aiming (QNX)" source={turret}
        hint="Run QNX with --preview-port 8080; enter the Pi's LAN address." />
    </section>

    <p className="note">Setup/debug only. Video stays on the local network and is never uploaded.</p>
  </main>;
}
