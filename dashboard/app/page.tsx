"use client";

import { useEffect, useState } from "react";

type Status = { active: boolean; reason: string };
const defaultUrl = "http://raspberrypi.local:8080";

export default function Home() {
  const [baseUrl, setBaseUrl] = useState(defaultUrl);
  const [savedUrl, setSavedUrl] = useState(defaultUrl);
  const [status, setStatus] = useState<Status>({ active: false, reason: "waiting for QNX" });
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    setBaseUrl(localStorage.getItem("medusa-preview-url") || defaultUrl);
  }, []);

  useEffect(() => {
    const poll = async () => {
      try {
        const response = await fetch(savedUrl + "/status.json", { cache: "no-store" });
        if (!response.ok) throw new Error();
        setStatus(await response.json());
        setConnected(true);
      } catch {
        setConnected(false);
        setStatus({ active: false, reason: "QNX preview unavailable" });
      }
    };
    poll();
    const timer = window.setInterval(poll, 750);
    return () => window.clearInterval(timer);
  }, [savedUrl]);

  const connect = () => {
    const normalized = baseUrl.replace(/\/$/, "");
    localStorage.setItem("medusa-preview-url", normalized);
    setSavedUrl(normalized);
  };

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
        {connected ? <img src={savedUrl + "/stream.mjpg"} alt="Fixed QNX camera preview" /> : <p>Camera preview will appear here.</p>}
      </div>
      <aside className={status.active ? "alert active" : "alert"}>
        <p className="eyebrow">CURRENT STATE</p>
        <strong>{status.active ? "SNOOPER DETECTED" : "CLEAR"}</strong>
        <p>{status.reason}</p>
      </aside>
    </section>

    <p className="note">Setup/debug only. Video stays on the local network and is never uploaded.</p>
  </main>;
}

