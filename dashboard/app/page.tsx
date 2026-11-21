"use client";

import { useEffect, useState } from "react";

type NoseOffset = { dx_px: number; dy_px: number; distance_px: number };
type Status = {
  active: boolean;
  reason: string;
  face_count?: number | null;
  snooper_detected?: boolean;
  snooper_count?: number;
  nose_offset?: NoseOffset | null;
};
const defaultUrl = "http://raspberrypi.local:8080";

function describeOffset(value: number, negative: string, positive: string) {
  const pixels = Math.round(Math.abs(value));
  return pixels ? `${pixels} px ${value < 0 ? negative : positive}` : "0 px";
}

export default function Home() {
  const [baseUrl, setBaseUrl] = useState(defaultUrl);
  const [savedUrl, setSavedUrl] = useState(defaultUrl);
  const [status, setStatus] = useState<Status>({ active: false, reason: "waiting for QNX" });
  const [connected, setConnected] = useState(false);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [previewAttempt, setPreviewAttempt] = useState(0);

  useEffect(() => {
    const rememberedUrl = localStorage.getItem("medusa-preview-url") || defaultUrl;
    setBaseUrl(rememberedUrl);
    setSavedUrl(rememberedUrl);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let activeRequest: AbortController | undefined;
    let timer: number;
    const poll = async () => {
      const request = new AbortController();
      activeRequest = request;
      const timeout = window.setTimeout(() => request.abort(), 3000);
      try {
        const response = await fetch(savedUrl + "/status.json", {
          cache: "no-store",
          signal: request.signal,
        });
        if (!response.ok) throw new Error();
        const currentStatus = await response.json();
        if (cancelled) return;
        setStatus(currentStatus);
        setConnected(true);
      } catch {
        if (cancelled) return;
        setConnected(false);
        setStatus({ active: false, reason: "QNX preview unavailable" });
      } finally {
        window.clearTimeout(timeout);
      }
      timer = window.setTimeout(poll, 750);
    };
    void poll();
    return () => {
      cancelled = true;
      activeRequest?.abort();
      window.clearTimeout(timer);
    };
  }, [savedUrl]);

  useEffect(() => {
    if (connected) {
      setPreviewFailed(false);
      setPreviewAttempt((attempt) => attempt + 1);
    }
  }, [connected]);

  useEffect(() => {
    if (!previewFailed) return;
    const timer = window.setTimeout(() => setPreviewFailed(false), 1500);
    return () => window.clearTimeout(timer);
  }, [previewFailed]);

  const connect = () => {
    const normalized = baseUrl.trim().replace(/\/$/, "");
    localStorage.setItem("medusa-preview-url", normalized);
    setSavedUrl(normalized);
    setPreviewFailed(false);
    setPreviewAttempt((attempt) => attempt + 1);
  };

  const snooperDetected = status.snooper_detected ?? status.active;
  const snooperCount = status.snooper_count ?? 0;
  const snooperLabel = snooperCount > 1 ? `${snooperCount} SNOOPERS DETECTED` : "SNOOPER DETECTED";
  const noseOffset = status.nose_offset;

  return <main>
    <header>
      <div><p className="eyebrow">LOCAL QNX SETUP VIEW</p><h1>Medusa</h1></div>
      <span className={connected ? "chip connected" : "chip"}>{connected ? "QNX connected" : "Waiting for QNX"}</span>
    </header>

    <section className="setup">
      <label htmlFor="pi-url">Pi preview address</label>
      <div><input id="pi-url" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} /><button onClick={connect}>Connect</button></div>
      <small>Run QNX with <code>--preview-port 8080</code>; enter the Pi's LAN address.</small>
    </section>

    <section className="grid">
      <div className="video">
        {previewFailed ? <p>Reconnecting camera preview…</p> :
          <img key={previewAttempt} src={savedUrl + "/stream.mjpg"} alt="Fixed QNX camera preview"
            onError={() => setPreviewFailed(true)} />}
      </div>
      <aside className={connected && snooperDetected ? "alert active" : "alert"}>
        <p className="eyebrow">CURRENT STATE</p>
        <strong>{connected ? (snooperDetected ? snooperLabel : "CLEAR") : "UNAVAILABLE"}</strong>
        {typeof status.face_count === "number" && (
          <p>{status.face_count} {status.face_count === 1 ? "face" : "faces"} detected</p>
        )}
        {connected && status.snooper_detected && (
          <section className="nose-offset" aria-label="Closest snooper nose offset">
            <p>Closest snooper's nose to preview center</p>
            {noseOffset ? (
              <>
                <strong>{Math.round(noseOffset.distance_px)} px</strong>
                <p>Horizontal {describeOffset(noseOffset.dx_px, "left", "right")}</p>
                <p>Vertical {describeOffset(noseOffset.dy_px, "up", "down")}</p>
              </>
            ) : <p>Nose position unavailable</p>}
          </section>
        )}
        {connected && <p>Privacy alert {status.active ? "active" : "inactive"}</p>}
        <small>{status.reason}</small>
      </aside>
    </section>

    <p className="note">Setup/debug only. Video stays on the local network and is never uploaded.</p>
  </main>;
}

