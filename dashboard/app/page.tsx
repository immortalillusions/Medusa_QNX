"use client";

import { useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------------------------
// Data shapes. Every field is optional on the wire; the page renders whatever is present.
// ---------------------------------------------------------------------------------------------

type Settings = { peeper_mode?: "one" | "two"; screen_switch?: boolean };

// /status.json from the laptop detector (:8081) and the Pi AI/turret camera (:8080).
type Status = {
  active: boolean;
  reason: string;
  face_count?: number | null;
  snooper_count?: number;
  mode?: string;
  fps?: number | null;
  // turret only
  laptop_active?: boolean | null;
  laptop_age_s?: number | null;
  laptop_target_x?: number | null;
  aim_target?: [number, number] | null;
  aim_match?: string | null;
  expected_x?: number | null;
  fallback_aim?: boolean;
  pan_state?: string;
  pan_angle?: number | null;
  pan_actual?: number | null;
  pan_error?: number | null;
  learned_points?: number | null;
  // laptop only
  peeper_id?: number | null;
  turret_on?: boolean;
  docs_enabled?: boolean;
  faces?: unknown[];
  settings?: Settings;
};

type SystemStats = {
  cpu_total_pct?: number | null;
  cpu_control_pct?: number | null;
  cpu_ai_pct?: number | null;
  mem_free_mb?: number | null;
  mem_total_mb?: number | null;
  temp_c?: number | null;
  uptime_s?: number | null;
  sampled_age_s?: number | null;
};

// Served by the high-priority control process on the Pi at :8082/control.json.
type Control = {
  state?: "SAFE" | "ACTIVE";
  safe_reason?: string;
  loop_hz?: number;
  period_ms_mean?: number;
  jitter_ms_p99?: number;
  period_ms_max?: number;
  missed_deadlines?: number;
  missed_deadlines_total?: number;
  target_age_s?: number;
  ai_alive?: boolean;
  arduino?: {
    link_ok?: boolean; armed?: boolean; parking?: boolean; gate_closed?: boolean | null;
    pan?: number; tilt?: number; age_s?: number;
  };
  stress?: { running?: boolean; seconds_left?: number };
  system?: SystemStats | null;
};

type Source = ReturnType<typeof useSource>;
type Tone = "ok" | "warn" | "bad" | "idle";

const CONTROL_PORT = 8082;
const SPARK_SAMPLES = 90;
const STATUS_POLL_MS = 220;
const CONTROL_POLL_MS = 220;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

function remember(key: string, value?: string) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {}
  return null;
}

const isNum = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);
const fmt = (n: unknown, digits: number, unit = "") => (isNum(n) ? `${n.toFixed(digits)}${unit}` : "—");

function fmtUptime(s: unknown) {
  if (!isNum(s)) return { value: "—", unit: "" };
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d > 0) return { value: `${d}d ${h}`, unit: "h" };
  if (h > 0) return { value: `${h}h ${m}`, unit: "m" };
  return { value: `${m}m ${Math.floor(s % 60)}`, unit: "s" };
}

function timeout(ms: number) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, done: () => window.clearTimeout(timer) };
}

// POST with an empty body; resolves to an error message or null on success.
async function post(url: string): Promise<string | null> {
  const t = timeout(3000);
  try {
    const response = await fetch(url, { method: "POST", signal: t.signal });
    return response.ok ? null : `refused (HTTP ${response.status})`;
  } catch {
    return "offline";
  } finally {
    t.done();
  }
}

// POST JSON. If the browser blocks the application/json preflight (server without OPTIONS),
// retry once as text/plain, which is a CORS "simple request" carrying the same JSON body.
async function postJson(url: string, body: unknown): Promise<string | null> {
  const send = async (type: string) => {
    const t = timeout(3000);
    try {
      return await fetch(url, { method: "POST", headers: { "Content-Type": type }, body: JSON.stringify(body), signal: t.signal });
    } finally {
      t.done();
    }
  };
  try {
    let response: Response;
    try {
      response = await send("application/json");
    } catch {
      response = await send("text/plain");
    }
    return response.ok ? null : `refused (HTTP ${response.status})`;
  } catch {
    return "offline";
  }
}

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
      const abortTimer = window.setTimeout(() => request?.abort(), timeoutMs);
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
        window.clearTimeout(abortTimer);
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
  const poll = usePoll<Status>(url, "/status.json", STATUS_POLL_MS, 1500);
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

  return { input, setInput, url, status, raw: poll.data, connected, previewFailed, setPreviewFailed, attempt, connect };
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

type ControlUrl = ReturnType<typeof useControlUrl>;
type ControlPoll = ReturnType<typeof usePoll<Control>>;

// Rolling history of a polled number.
function useHistory(sample: unknown, tick: unknown) {
  const [history, setHistory] = useState<number[]>([]);
  useEffect(() => {
    if (isNum(sample)) setHistory((h) => [...h.slice(-(SPARK_SAMPLES - 1)), sample]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);
  return history;
}

// Detection latencies, measured client-side from the polled JSON (so approximate: ±1 poll).
// t0 is the laptop's `active` rising edge as the page first sees it.
type Latency = { ms: number; sameFrame?: boolean; n: number } | null;
function useLatencies(laptop: Source, turret: Source) {
  const riseAt = useRef<number | null>(null);
  const prevActive = useRef(false);
  const waitDocs = useRef(false);
  const waitAim = useRef(false);
  const [docs, setDocs] = useState<Latency>(null);
  const [aim, setAim] = useState<Latency>(null);

  const lActive = laptop.connected && laptop.status.active;
  const lDocs = laptop.connected && laptop.status.docs_enabled === true;
  const tActive = turret.connected && turret.status.active;

  useEffect(() => {
    const now = performance.now();
    if (lActive && !prevActive.current) {
      riseAt.current = now;
      waitDocs.current = true;
      // A turret that was already aiming before the laptop fired is not a measurement.
      waitAim.current = !tActive;
    }
    prevActive.current = lActive;
    const t0 = riseAt.current;
    if (t0 === null) return;
    if (now - t0 > 5000) {
      waitDocs.current = false;
      waitAim.current = false;
      return;
    }
    if (waitDocs.current && lActive && lDocs) {
      waitDocs.current = false;
      const ms = now - t0;
      setDocs((d) => ({ ms, sameFrame: ms < 1, n: (d?.n ?? 0) + 1 }));
    }
    if (waitAim.current && tActive) {
      waitAim.current = false;
      const ms = now - t0;
      setAim((a) => ({ ms, n: (a?.n ?? 0) + 1 }));
    }
  }, [lActive, lDocs, tActive]);

  return { docs, aim };
}

// ---------------------------------------------------------------------------------------------
// Small components
// ---------------------------------------------------------------------------------------------

function Spinner() {
  return <span className="spinner" aria-hidden />;
}

function Sparkline({ values, nominal, floor, tone, label }: {
  values: number[]; nominal?: number; floor?: number; tone: "accent" | "muted"; label: string;
}) {
  const width = 600, height = 72;
  const top = Math.max(1, floor ?? 0, ...values, isNum(nominal) ? nominal * 1.5 : 0) * 1.08;
  const y = (v: number) => height - (v / top) * height;
  const offset = SPARK_SAMPLES - values.length;
  const points = values.map((v, i) => `${((offset + i) * width) / (SPARK_SAMPLES - 1)},${y(v).toFixed(1)}`).join(" ");
  return <svg className="spark" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label={label}>
    {isNum(nominal) && <line x1={0} x2={width} y1={y(nominal)} y2={y(nominal)} className="spark-nominal" />}
    {values.length > 1 && <polyline points={points} className={`spark-line ${tone}`} />}
  </svg>;
}

function Tile({ label, value, unit, sub, tone = "idle", children }: {
  label: string; value: string; unit?: string; sub?: React.ReactNode; tone?: Tone; children?: React.ReactNode;
}) {
  const empty = value === "—";
  return <div className={`tile tone-${tone}`}>
    <p className="tile-label">{label}</p>
    <p className={empty ? "tile-value empty" : "tile-value"}>
      {value}{!empty && unit && <span className="tile-unit">{unit}</span>}
    </p>
    {children}
    {sub && <p className="tile-sub">{sub}</p>}
  </div>;
}

function HealthChip({ label, tone, value }: { label: string; tone: Tone; value: string }) {
  return <span className="chip">
    <i className={`dot ${tone}`} aria-hidden />
    <span className="chip-label">{label}</span>
    <span className="chip-value">{value}</span>
  </span>;
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
  if (status.pan_state) lines.push(`Pan state: ${status.pan_state}`);
  if (status.pan_state === "fallback" || status.fallback_aim) lines.push("Aim: FALLBACK (laptop)");
  if (isNum(status.learned_points)) lines.push(`Learned map: ${status.learned_points} points`);
  if (status.docs_enabled !== undefined) lines.push(`Docs switch allowed: ${status.docs_enabled ? "yes" : "no"}`);
  if (status.turret_on !== undefined) lines.push(`Turret on (laptop view): ${status.turret_on ? "yes" : "no"}`);
  return lines;
}

// ---------------------------------------------------------------------------------------------
// Top bar: state pill, arm button, laptop settings
// ---------------------------------------------------------------------------------------------

function useArm(c: Control | null, controlUrl: ControlUrl) {
  const arduino = c?.arduino;
  const armed = arduino?.armed === true;
  const parking = arduino?.parking === true;
  const [sending, setSending] = useState(false);
  // After a 202 the button waits for control.json to report the new state (never optimistic).
  const [awaiting, setAwaiting] = useState<{ target: boolean; until: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!awaiting) return;
    if (armed === awaiting.target || (!awaiting.target && parking)) { setAwaiting(null); return; }
    const timer = window.setTimeout(() => setAwaiting(null), Math.max(0, awaiting.until - Date.now()));
    return () => window.clearTimeout(timer);
  }, [awaiting, armed, parking]);

  useEffect(() => {
    if (!error) return;
    const timer = window.setTimeout(() => setError(null), 5000);
    return () => window.clearTimeout(timer);
  }, [error]);

  // POST /arm or /disarm queues an A line; the firmware decides (kill gate, heartbeat) and the
  // button only ever shows what control.json's arduino fields report.
  const blocked = !c ? "control loop offline"
    : arduino?.link_ok !== true ? "Arduino link down"
    : arduino.gate_closed === false ? "kill gate open: insert the tab to arm"
    : parking ? "gliding home, then limp; arm again once it stops"
    : null;
  const pending = sending || awaiting !== null;
  const label = sending ? "Sending…" : awaiting ? (awaiting.target ? "Arming…" : "Switching off…")
    : parking ? "Returning home…" : armed ? "Switch off" : "Arm turret";
  const toggle = async () => {
    const target = !armed;
    setSending(true);
    setError(null);
    const failure = await post(`${controlUrl.url}/${armed ? "disarm" : "arm"}`);
    setSending(false);
    if (failure) setError(`control loop ${failure}`);
    else setAwaiting({ target, until: Date.now() + 3000 });
  };
  return { armed, parking, blocked, pending, label, error, toggle };
}

function useLaptopSettings(laptop: Source) {
  const settings = laptop.connected ? laptop.status.settings : undefined;
  const [pending, setPending] = useState<{ patch: Settings; until: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!pending) return;
    const landed = Object.entries(pending.patch).every(([k, v]) => settings?.[k as keyof Settings] === v);
    if (landed) { setPending(null); return; }
    const timer = window.setTimeout(() => {
      setPending(null);
      setError("laptop detector did not apply the change");
    }, Math.max(0, pending.until - Date.now()));
    return () => window.clearTimeout(timer);
  }, [pending, settings]);

  useEffect(() => {
    if (!error) return;
    const timer = window.setTimeout(() => setError(null), 5000);
    return () => window.clearTimeout(timer);
  }, [error]);

  const unavailable = !laptop.connected ? "laptop detector offline"
    : !settings ? "laptop detector needs restart" : null;

  const send = async (patch: Settings) => {
    setError(null);
    setPending({ patch, until: Date.now() + 3500 });
    const failure = await postJson(`${laptop.url}/settings`, patch);
    if (failure) {
      setPending(null);
      setError(`laptop detector ${failure}`);
    }
  };
  return { settings, unavailable, pending: pending?.patch ?? null, error, send };
}

function TopBar({ laptop, turret, control, controlUrl }: {
  laptop: Source; turret: Source; control: ControlPoll; controlUrl: ControlUrl;
}) {
  const c = control.connected ? control.data ?? {} : null;
  const arm = useArm(c, controlUrl);
  const s = useLaptopSettings(laptop);

  const peeper = laptop.connected && laptop.status.active;
  const state = peeper ? "PEEPER" : arm.armed ? "ARMED" : "SAFE";
  const reason = peeper
    ? `${laptop.status.reason}${laptop.status.docs_enabled ? " · docs switched" : ""}${turret.connected && turret.status.active ? " · turret aiming" : ""}`
    : arm.armed ? (c?.safe_reason ?? "turret armed, waiting for a peeper")
    : !c ? "control loop offline: turret cannot move"
    : c.safe_reason ?? "turret switched off";
  const secondary = peeper ? null
    : laptop.connected ? `Laptop: ${laptop.status.reason}`
    : "Laptop detector offline";

  // Health
  const laptopFps = laptop.connected ? laptop.status.fps : undefined;
  const turretFps = turret.connected ? turret.status.fps : undefined;
  const linkAge = turret.connected ? turret.status.laptop_age_s : undefined;
  const arduino = c?.arduino;

  const modeValue = s.settings?.peeper_mode;
  const switchValue = s.settings?.screen_switch;
  const disabled = !!s.unavailable || s.pending !== null;

  return <header className="topbar">
    <div className="topbar-main">
      <div className="brand">
        <span className="brand-mark" aria-hidden><i /></span>
        <div>
          <h1>Medusa</h1>
          <p>Shoulder-surfer guard · QNX real-time turret</p>
        </div>
      </div>

      <div className="state" aria-live="polite">
        <span className={`pill pill-${state.toLowerCase()}`}>{state}</span>
        <div className="state-text">
          <p className="state-reason">{reason}</p>
          {secondary && <p className="state-sub">{secondary}</p>}
        </div>
      </div>

      <div className="arm">
        <button className={`btn btn-arm${arm.armed ? " is-armed" : ""}`} onClick={arm.toggle}
          disabled={!!arm.blocked || arm.pending}>
          {arm.pending && <Spinner />}{arm.label}
        </button>
        {(arm.blocked || arm.error) && <p className={arm.error ? "note note-bad" : "note"}>{arm.blocked ?? arm.error}</p>}
      </div>
    </div>

    <div className="toolbar">
      <div className="field">
        <span className="field-label" id="peeper-mode-label">Peeper mode</span>
        <div className="seg" role="radiogroup" aria-labelledby="peeper-mode-label">
          {([["one", "One person"], ["two", "Two people"]] as const).map(([v, text]) =>
            <button key={v} role="radio" aria-checked={modeValue === v} disabled={disabled}
              className={modeValue === v ? "seg-on" : undefined}
              onClick={() => modeValue !== v && s.send({ peeper_mode: v })}>
              {s.pending?.peeper_mode === v && <Spinner />}{text}
            </button>)}
        </div>
      </div>

      <div className="field">
        <span className="field-label" id="screen-switch-label">Screen switching</span>
        <button role="switch" aria-checked={switchValue === true} aria-labelledby="screen-switch-label"
          className={`toggle${switchValue === true ? " toggle-on" : ""}`} disabled={disabled}
          onClick={() => s.send({ screen_switch: !(switchValue === true) })}>
          <span className="toggle-track"><span className="toggle-knob" /></span>
          <span className="toggle-text">
            {s.pending?.screen_switch !== undefined && <Spinner />}
            {switchValue === undefined ? "—" : switchValue ? "On" : "Off"}
          </span>
        </button>
      </div>

      {(s.unavailable || s.error) && <p className={s.error ? "note note-bad" : "note"}>{s.error ?? s.unavailable}</p>}

      <div className="health">
        <HealthChip label="Laptop" tone={laptop.connected ? "ok" : "bad"}
          value={laptop.connected ? fmt(laptopFps, 0, " fps") : "offline"} />
        <HealthChip label="Laptop → Pi" tone={!isNum(linkAge) ? "idle" : linkAge < 0.5 ? "ok" : "bad"}
          value={isNum(linkAge) ? `${linkAge.toFixed(2)} s` : "no data"} />
        <HealthChip label="Turret cam" tone={!turret.connected ? "bad" : !isNum(turretFps) || turretFps > 0 ? "ok" : "warn"}
          value={turret.connected ? fmt(turretFps, 0, " fps") : "offline"} />
        <HealthChip label="AI" tone={c?.ai_alive === true ? "ok" : c?.ai_alive === false ? "bad" : "idle"}
          value={!c ? "unknown" : c.ai_alive === undefined ? "no data" : c.ai_alive ? "alive" : "stalled"} />
        <HealthChip label="Control" tone={c ? "ok" : "bad"} value={c ? fmt(c.loop_hz, 0, " Hz") : "offline"} />
        <HealthChip label="Arduino" tone={arduino?.link_ok === true ? "ok" : arduino?.link_ok === false ? "bad" : "idle"}
          value={!arduino ? (c ? "no data" : "unknown")
            : `${arduino.armed ? "armed" : "disarmed"}${isNum(arduino.pan) ? ` · pan ${arduino.pan.toFixed(0)}°` : ""}${isNum(arduino.tilt) ? ` tilt ${arduino.tilt.toFixed(0)}°` : ""}`} />
      </div>
    </div>
  </header>;
}

// ---------------------------------------------------------------------------------------------
// Cameras
// ---------------------------------------------------------------------------------------------

function CameraPanel({ title, sub, source, ratio, badge, caption, offlineHint }: {
  title: string; sub: string; source: Source; ratio: "wide" | "classic";
  badge: { text: string; tone: "bad" | "warn" } | null; caption: React.ReactNode; offlineHint: string;
}) {
  const { connected } = source;
  const showStream = !source.previewFailed;
  return <section className="panel camera">
    <div className="panel-head">
      <div>
        <h2>{title}</h2>
        <p className="panel-sub">{sub}</p>
      </div>
      <span className={connected ? "live live-on" : "live"}><i aria-hidden />{connected ? "Live" : "Offline"}</span>
    </div>
    <div className={`media media-${ratio}`}>
      {showStream && <img key={source.attempt} src={source.url + "/stream.mjpg"} alt={`${title} stream`}
        onError={() => source.setPreviewFailed(true)} />}
      {!showStream && <div className="media-empty">
        <p>{connected ? "Reconnecting stream…" : "Camera offline"}</p>
        <small>{offlineHint}</small>
        <small className="mono">{source.url || "(no address)"}/stream.mjpg</small>
      </div>}
      {badge && <span className={`media-badge badge-${badge.tone}`}>{badge.text}</span>}
    </div>
    <div className="caption">{connected ? caption : <span className="muted">No status from {source.url || "(no address)"}</span>}</div>
  </section>;
}

function Cap({ k, v }: { k: string; v: React.ReactNode }) {
  return <span className="cap"><span className="cap-k">{k}</span><span className="cap-v">{v}</span></span>;
}

// ---------------------------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------------------------

function Metrics({ laptop, turret, control, latency }: {
  laptop: Source; turret: Source; control: ControlPoll; latency: ReturnType<typeof useLatencies>;
}) {
  const c = control.connected ? control.data ?? {} : null;
  const sys = c?.system ?? null;
  const t = turret.connected ? turret.status : null;
  const l = laptop.connected ? laptop.status : null;

  const laptopFps = l?.fps;
  const turretFps = t?.fps;
  const frameMs = isNum(laptopFps) && laptopFps > 0 ? 1000 / laptopFps : undefined;

  const docs = latency.docs;
  const docsValue = !docs ? "—" : docs.sameFrame ? (isNum(frameMs) ? `≈${frameMs.toFixed(0)}` : "<1") : docs.ms.toFixed(0);
  const docsUnit = docs?.sameFrame && !isNum(frameMs) ? "frame" : "ms";
  const docsSub = !docs
    ? (l?.settings?.screen_switch === false ? "screen switching is off" : "approx, polled · waiting for a peeper")
    : docs.sameFrame ? "approx, polled · same frame as detection" : `approx, polled · ${docs.n} event${docs.n === 1 ? "" : "s"}`;

  const aim = latency.aim;
  const aimValue = !aim ? "—" : aim.ms >= 1000 ? (aim.ms / 1000).toFixed(2) : aim.ms.toFixed(0);
  const aimUnit = aim && aim.ms >= 1000 ? "s" : "ms";

  const panError = t?.pan_error;
  const errTone: Tone = !isNum(panError) ? "idle" : Math.abs(panError) > 8 ? "warn" : "ok";

  const jitter = c?.jitter_ms_p99;
  const missed = c?.missed_deadlines;
  const nominal = isNum(c?.loop_hz) && c.loop_hz > 0 ? 1000 / c.loop_hz : undefined;

  const cpu = sys?.cpu_total_pct;
  const ctl = sys?.cpu_control_pct;
  const ai = sys?.cpu_ai_pct;
  const memTotal = sys?.mem_total_mb;
  const memUsed = isNum(memTotal) && isNum(sys?.mem_free_mb) ? memTotal - sys.mem_free_mb : undefined;
  const memPct = isNum(memUsed) && isNum(memTotal) && memTotal > 0 ? (memUsed / memTotal) * 100 : undefined;
  const temp = sys?.temp_c;
  const uptime = fmtUptime(sys?.uptime_s);
  const sysNote = !c ? "control loop offline" : !sys ? "needs control.json system stats" : undefined;

  const bar = (v: unknown) => (isNum(v) ? `${Math.max(0, Math.min(100, v))}%` : "0%");
  // Process shares may be per-core percentages (AI on four cores can read 200%+) while the total
  // is machine-wide; squeeze the split into the total so the bar never overflows.
  const split = (isNum(ctl) ? ctl : 0) + (isNum(ai) ? ai : 0);
  const squeeze = isNum(cpu) && split > cpu && split > 0 ? cpu / split : 1;
  const perCore = (isNum(ctl) && ctl > 100) || (isNum(ai) && ai > 100);

  return <section className="metrics" aria-label="Demo metrics">
    <Tile label="Detection → docs switch" value={docsValue} unit={docsUnit} sub={docsSub} tone={docs ? "ok" : "idle"} />
    <Tile label="Detection → turret aiming" value={aimValue} unit={aimUnit} tone={aim ? "ok" : "idle"}
      sub={aim ? `approx, polled · ${aim.n} event${aim.n === 1 ? "" : "s"}` : "approx, polled · waiting for a peeper"} />
    <Tile label="Centring error" value={isNum(panError) ? Math.abs(panError).toFixed(1) : "—"} unit="°" tone={errTone}
      sub={isNum(panError) ? `pan ${fmt(t?.pan_angle, 1, "°")} · signed ${panError.toFixed(1)}°` : t ? "no target" : "turret offline"} />
    <Tile label="Laptop detector" value={fmt(laptopFps, 1)} unit="fps" tone={!l ? "idle" : "ok"}
      sub={l ? `${l.face_count ?? 0} face${l.face_count === 1 ? "" : "s"} in view` : "offline"} />
    <Tile label="Turret AI (QNX)" value={fmt(turretFps, 1)} unit="fps"
      tone={!t ? "idle" : isNum(turretFps) && turretFps < 5 ? "warn" : "ok"}
      sub={t ? `${t.face_count ?? 0} face${t.face_count === 1 ? "" : "s"} · ${t.mode ?? "mode ?"}` : "offline"} />
    <Tile label="Control loop period" value={fmt(c?.period_ms_mean, 2)} unit="ms" tone={c ? "ok" : "idle"}
      sub={c ? `target ${fmt(nominal, 1, " ms")} · ${fmt(c.loop_hz, 1, " Hz")}` : "control loop offline"} />
    <Tile label="p99 jitter" value={fmt(jitter, 2)} unit="ms"
      tone={!isNum(jitter) ? "idle" : jitter > 5 ? "warn" : "ok"}
      sub={c ? `worst period ${fmt(c.period_ms_max, 1, " ms")}` : "control loop offline"} />
    <Tile label="Missed deadlines" value={fmt(missed, 0)}
      tone={!isNum(missed) ? "idle" : missed > 0 ? "warn" : "ok"}
      sub={c ? (isNum(c.missed_deadlines_total) ? `recent window · ${c.missed_deadlines_total} since boot` : "recent window") : "control loop offline"} />
    <Tile label="Pi CPU" value={fmt(cpu, 0)} unit="%" tone={!isNum(cpu) ? "idle" : cpu > 90 ? "warn" : "ok"}
      sub={sysNote ?? `control ${fmt(ctl, 0, "%")} · AI ${fmt(ai, 0, "%")}${perCore ? " (per core)" : ""}`}>
      <div className="cpu-bar" aria-hidden>
        <span className="cpu-ctl" style={{ width: bar(isNum(ctl) ? ctl * squeeze : 0) }} />
        <span className="cpu-ai" style={{ width: bar(isNum(ai) ? ai * squeeze : 0) }} />
        <span className="cpu-rest" style={{ width: bar(isNum(cpu) ? cpu - split * squeeze : 0) }} />
      </div>
    </Tile>
    <Tile label="Memory used" value={isNum(memUsed) ? (memUsed / 1024).toFixed(2) : "—"} unit="GB"
      tone={!isNum(memPct) ? "idle" : memPct > 90 ? "warn" : "ok"}
      sub={sysNote ?? `of ${fmt(isNum(memTotal) ? memTotal / 1024 : undefined, 1, " GB")} · ${fmt(memPct, 0, "%")}`} />
    <Tile label="SoC temperature" value={fmt(temp, 1)} unit="°C"
      tone={!isNum(temp) ? "idle" : temp > 85 ? "bad" : temp > 75 ? "warn" : "ok"}
      sub={sysNote ?? (isNum(temp) && temp > 75 ? "running hot" : "Raspberry Pi SoC")} />
    <Tile label="Pi uptime" value={uptime.value} unit={uptime.unit} tone={isNum(sys?.uptime_s) ? "ok" : "idle"}
      sub={sysNote ?? "QNX on Raspberry Pi"} />
  </section>;
}

// ---------------------------------------------------------------------------------------------
// Real-time: stress test and sparklines
// ---------------------------------------------------------------------------------------------

function RealTime({ turret, control, controlUrl }: { turret: Source; control: ControlPoll; controlUrl: ControlUrl }) {
  const c = control.connected ? control.data ?? {} : null;
  const [stressPending, setStressPending] = useState(false);
  const [stressError, setStressError] = useState<string | null>(null);

  const sparkMetric = isNum(c?.period_ms_max) ? "period max" : "p99 jitter";
  const periods = useHistory(control.data?.period_ms_max ?? control.data?.jitter_ms_p99, control.data);
  const turretFps = turret.connected ? turret.status.fps : undefined;
  const fpsHistory = useHistory(turretFps, turret.raw);
  const nominal = c?.period_ms_mean ?? (isNum(c?.loop_hz) && c.loop_hz > 0 ? 1000 / c.loop_hz : undefined);

  useEffect(() => {
    if (!stressError) return;
    const timer = window.setTimeout(() => setStressError(null), 5000);
    return () => window.clearTimeout(timer);
  }, [stressError]);

  const stressRunning = c?.stress?.running === true;
  const runStress = async () => {
    setStressPending(true);
    setStressError(null);
    const failure = await post(`${controlUrl.url}/stress?seconds=20`);
    if (failure) setStressError(`control loop ${failure}`);
    setStressPending(false);
  };
  const stressNote = stressRunning ? `${fmt(c?.stress?.seconds_left, 0, " s")} left`
    : !c ? "control loop offline" : stressError;

  return <section className={`panel realtime${stressRunning ? " is-stressed" : ""}`}>
    <div className="rt-head">
      <div>
        <h2>Real-time under load</h2>
        <p className="rt-caption">AI slows down, control loop doesn&apos;t.</p>
      </div>
      <div className="rt-stress">
        <button className="btn btn-secondary" onClick={runStress} disabled={!c || stressRunning || stressPending}>
          {(stressPending || stressRunning) && <Spinner />}
          {stressRunning ? "Stress running…" : stressPending ? "Starting…" : "Run CPU stress test (20 s)"}
        </button>
        {stressNote && <p className={stressError ? "note note-bad" : "note"}>{stressNote}</p>}
      </div>
    </div>
    <div className="rt-charts">
      <div className="rt-chart">
        <div className="rt-chart-head">
          <span>Control loop {sparkMetric}</span>
          <span className="mono">{periods.length ? `${periods[periods.length - 1].toFixed(2)} ms` : "—"}</span>
        </div>
        <Sparkline values={periods} nominal={nominal} tone="accent" label="Recent control loop timing" />
        <p className="rt-chart-foot">dashed line: mean period {fmt(nominal, 2, " ms")} · last {SPARK_SAMPLES} samples</p>
      </div>
      <div className="rt-chart">
        <div className="rt-chart-head">
          <span>Turret AI frame rate</span>
          <span className="mono">{fpsHistory.length ? `${fpsHistory[fpsHistory.length - 1].toFixed(1)} fps` : "—"}</span>
        </div>
        <Sparkline values={fpsHistory} floor={20} tone="muted" label="Recent turret AI frame rate" />
        <p className="rt-chart-foot">best effort, lower priority · last {SPARK_SAMPLES} samples</p>
      </div>
    </div>
  </section>;
}

// ---------------------------------------------------------------------------------------------
// Advanced
// ---------------------------------------------------------------------------------------------

function AddressField({ id, label, source, hint }: { id: string; label: string; source: Source; hint: string }) {
  return <div className="addr">
    <label htmlFor={id}>{label}</label>
    <div className="addr-row">
      <input id={id} value={source.input} spellCheck={false} onChange={(e) => source.setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && source.connect()} />
      <button className="btn btn-secondary" onClick={source.connect}>Connect</button>
    </div>
    <small>{source.connected ? "Connected. " : "Not connected. "}{hint}</small>
  </div>;
}

function Advanced({ laptop, turret, control, controlUrl }: {
  laptop: Source; turret: Source; control: ControlPoll; controlUrl: ControlUrl;
}) {
  return <details className="panel advanced">
    <summary>Advanced: addresses and raw status</summary>
    <div className="adv-grid">
      <AddressField id="laptop-url" label="Laptop detector" source={laptop}
        hint="The Mac detector serves /stream.mjpg, /status.json and /settings." />
      <AddressField id="turret-url" label="Turret camera (Pi)" source={turret}
        hint="Run QNX with --preview-port 8080; enter the Pi's LAN address." />
      <div className="addr">
        <label htmlFor="control-url">Control loop (Pi)</label>
        <div className="addr-row">
          <input id="control-url" value={controlUrl.input} spellCheck={false}
            placeholder={controlUrl.derived || "http://<pi-host>:8082"}
            onChange={(e) => controlUrl.setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && controlUrl.save()} />
          <button className="btn btn-secondary" onClick={controlUrl.save}>Save</button>
          {controlUrl.override && <button className="btn btn-ghost" onClick={controlUrl.reset}>Reset</button>}
        </div>
        <small>{controlUrl.override ? "Override in use." : `Derived from the turret address (port ${CONTROL_PORT}).`} Polling {controlUrl.url || "nothing"}/control.json</small>
      </div>
    </div>
    <div className="adv-grid">
      {([["Laptop detector", laptop], ["Turret camera", turret]] as const).map(([name, src]) =>
        <div key={name} className="raw">
          <h3>{name}: {src.connected ? (src.status.active ? "PEEPER" : "clear") : "unavailable"}</h3>
          <ul>{src.connected && details(src.status).map((line) => <li key={line.split(":")[0]}>{line}</li>)}
            <li className="muted">{src.status.reason}</li></ul>
          {src.raw && <pre>{JSON.stringify(src.raw, null, 1)}</pre>}
        </div>)}
      <div className="raw">
        <h3>Control loop: {control.connected ? control.data?.state ?? "unknown" : "offline"}</h3>
        {control.data ? <pre>{JSON.stringify(control.data, null, 1)}</pre> : <ul><li className="muted">no answer</li></ul>}
      </div>
    </div>
    <p className="privacy">Video stays on the local network and is never uploaded.</p>
  </details>;
}

// ---------------------------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------------------------

export default function Home() {
  const laptop = useSource("medusa-laptop-url", "http://localhost:8081");
  const turret = useSource("medusa-turret-url", "http://172.20.10.2:8080");
  const controlUrl = useControlUrl(turret.url);
  const control = usePoll<Control>(controlUrl.url, "/control.json", CONTROL_POLL_MS, 1000);
  const latency = useLatencies(laptop, turret);

  const l = laptop.status, t = turret.status;
  const laptopCaption = <>
    <Cap k="fps" v={fmt(l.fps, 1)} />
    <Cap k="faces" v={isNum(l.face_count) ? l.face_count : "—"} />
    {isNum(l.peeper_id) && <Cap k="peeper" v={`#${l.peeper_id}`} />}
    <span className="cap-reason">{l.reason}</span>
  </>;
  const turretCaption = <>
    <Cap k="fps" v={fmt(t.fps, 1)} />
    <Cap k="aim" v={t.fallback_aim ? `${t.aim_match ?? "—"} (fallback)` : t.aim_match ?? "—"} />
    <Cap k="pan" v={fmt(t.pan_angle ?? t.pan_actual, 1, "°")} />
    <Cap k="error" v={isNum(t.pan_error) ? `${t.pan_error.toFixed(1)}°` : "—"} />
    <span className="cap-reason">{t.reason}</span>
  </>;

  return <main>
    <TopBar laptop={laptop} turret={turret} control={control} controlUrl={controlUrl} />

    <div className="cameras">
      <CameraPanel title="Laptop camera — who's looking" sub="Detector on the Mac · gaze per face" source={laptop}
        ratio="wide" caption={laptopCaption} offlineHint="Start the laptop detector, then check its address in Advanced."
        badge={laptop.connected && l.active ? { text: "PEEPER", tone: "bad" } : null} />
      <CameraPanel title="Turret camera — aiming (QNX)" sub="Raspberry Pi on QNX · face lock and pan" source={turret}
        ratio="classic" caption={turretCaption} offlineHint="Check the Pi is on the network and QNX preview is running."
        badge={!turret.connected ? null
          : t.pan_state === "fallback" ? { text: "Following from laptop", tone: "warn" }
          : t.active ? { text: "AIMING", tone: "warn" } : null} />
    </div>

    <Metrics laptop={laptop} turret={turret} control={control} latency={latency} />

    <RealTime turret={turret} control={control} controlUrl={controlUrl} />

    <Advanced laptop={laptop} turret={turret} control={control} controlUrl={controlUrl} />
  </main>;
}
