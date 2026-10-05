import { useEffect, useRef, useState, type ReactNode } from "react";
import { api, url } from "./api";
import { connectSocket } from "./realtime";
import { useApp } from "./store";
import type { ActionResult } from "./types";
export function Panel({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {children}
    </section>
  );
}
export function Badge({ children }: { children: ReactNode }) {
  return <span className="badge">{children}</span>;
}
export const display = (value: unknown): string =>
  value == null
    ? "Unavailable"
    : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);
export function Result({ result }: { result: ActionResult | null }) {
  if (!result) return null;
  const data = result.result || result;
  const failed =
    data.success === false ||
    [
      "failed",
      "unavailable",
      "not_configured",
      "gps_unavailable",
      "unknown_destination",
    ].includes(String(data.status).toLowerCase());
  return (
    <div className={failed ? "result error" : "result"} role="status">
      <strong>
        {data.status ||
          (data.has_text === false || data.has_result === false
            ? "No result"
            : "Backend response")}
      </strong>
      <p>
        {data.text || data.message || result.message || "Request processed."}
      </p>
      {data.color && <p>Color: {data.color}</p>}
      {typeof data.confidence === "number" && !data.intent && (
        <p>Confidence: {Math.round(data.confidence * 100)}%</p>
      )}
    </div>
  );
}
export function Action({
  label,
  run,
  danger = false,
}: {
  label: string;
  run: () => Promise<ActionResult>;
  danger?: boolean;
}) {
  const [busy, setBusy] = useState(false),
    [result, setResult] = useState<ActionResult | null>(null),
    [error, setError] = useState("");
  return (
    <div>
      <button
        className={danger ? "danger" : ""}
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          setError("");
          setResult(null);
          try {
            setResult(await run());
          } catch (e) {
            setError((e as Error).message);
          } finally {
            setBusy(false);
          }
        }}
      >
        {busy ? "Processing…" : label}
      </button>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <Result result={result} />
    </div>
  );
}
export function LiveFeed() {
  const image = useRef<HTMLImageElement>(null);
  const [state, setState] = useState("Connecting");
  const [fresh, setFresh] = useState(false);
  const camera = useApp((s) => s.status?.camera_connected),
    source = useApp((s) => s.status?.camera_source);
  useEffect(() => {
    let current = "",
      last = 0;
    const close = connectSocket(
      "/ws/stream",
      (e) => {
        if (!(e.data instanceof Blob)) return;
        const next = URL.createObjectURL(e.data);
        if (image.current) image.current.src = next;
        if (current) URL.revokeObjectURL(current);
        current = next;
        last = Date.now();
        setFresh(true);
      },
      setState,
    );
    const timer = setInterval(() => {
      if (Date.now() - last > 5000) setFresh(false);
    }, 1000);
    return () => {
      close();
      clearInterval(timer);
      if (current) URL.revokeObjectURL(current);
    };
  }, []);
  return (
    <div className="video">
      <img
        ref={image}
        alt="Backend camera stream with detection overlays"
        hidden={!fresh || !camera}
      />
      <span className="video-tag">
        {source || "Camera"} · {state}
      </span>
      {(!fresh || !camera) && (
        <div className="video-empty">
          {camera ? "Waiting for camera frames…" : "Camera unavailable"}
          <small>Detections and overlays come from VisionMate.</small>
        </div>
      )}
    </div>
  );
}
export function MapView() {
  const location = useApp((s) => s.location),
    nav = useApp((s) => s.navigation);
  const [src, setSrc] = useState(""),
    [error, setError] = useState("");
  const lat = location?.lat,
    lon = location?.lon,
    route = nav?.route?.polyline;
  useEffect(() => {
    if (lat == null || lon == null) {
      setSrc("");
      return;
    }
    let disposed = false,
      objectURL = "";
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    setError("");
    setSrc("");
    fetch(url("/api/navigation/map"), { signal: controller.signal })
      .then(async (r) => {
        if (!r.ok) {
          const e = await r.json();
          throw new Error(e.detail || "Map unavailable");
        }
        return r.blob();
      })
      .then((blob) => {
        if (!disposed) {
          objectURL = URL.createObjectURL(blob);
          setSrc(objectURL);
        }
      })
      .catch((e) => {
        if (!disposed) setError(e.message);
      });
    return () => {
      disposed = true;
      controller.abort();
      clearTimeout(timeout);
      if (objectURL) URL.revokeObjectURL(objectURL);
    };
  }, [lat, lon, route]);
  if (lat == null || lon == null)
    return <div className="empty">Location unavailable.</div>;
  return (
    <div>
      {src ? (
        <img
          className="map"
          src={src}
          alt="Google map showing current position and available destination and route"
        />
      ) : (
        <p className={error ? "error" : "muted"}>
          {error || "Loading Google map…"}
        </p>
      )}
      <p>
        {lat.toFixed(5)}, {lon.toFixed(5)} · Accuracy{" "}
        {location?.accuracy == null
          ? "unavailable"
          : `${Math.round(location.accuracy)} m`}
      </p>
      <p className="muted">
        {location?.source === "mock" ? "Demo location" : location?.source} ·{" "}
        {location?.timestamp
          ? new Date(location.timestamp * 1000).toLocaleString()
          : "No location update"}
      </p>
    </div>
  );
}
export function DeviceStatus() {
  const health = useApp((s) => s.health),
    metrics = useApp((s) => s.metrics);
  return (
    <>
      <Panel title="Subsystems">
        <div className="system-grid">
          {Object.entries(health?.subsystems || {})
            .filter(([, v]) => typeof v === "object" && v !== null)
            .map(([name, data]) => {
              const d = data as Record<string, unknown>;
              return (
                <div className="system" key={name}>
                  <h3>{name.replaceAll("_", " ")}</h3>
                  <Badge>
                    {display(
                      d.status ??
                        d.connection_state ??
                        (typeof d.ready === "boolean"
                          ? d.ready
                            ? "READY"
                            : "UNAVAILABLE"
                          : typeof d.loaded === "boolean"
                            ? d.loaded
                              ? "READY"
                              : "UNAVAILABLE"
                            : (d.state ?? "Reported")),
                    )}
                  </Badge>
                  <dl>
                    {Object.entries(d)
                      .filter(
                        ([k, v]) =>
                          [
                            "provider",
                            "active_source",
                            "device",
                            "model",
                            "loaded",
                            "engine_loaded",
                            "connected",
                            "mode",
                            "error",
                          ].includes(k) && v != null,
                      )
                      .map(([k, v]) => (
                        <div key={k}>
                          <dt>{k.replaceAll("_", " ")}</dt>
                          <dd>{display(v)}</dd>
                        </div>
                      ))}
                  </dl>
                </div>
              );
            })}
        </div>
        {!health && <p>Waiting for backend health.</p>}
      </Panel>
      <Panel title="Performance">
        <dl className="metrics">
          {[
            "effective_fps",
            "yolo_latency_ms",
            "frame_age_ms",
            "dropped_frames",
            "process_ram_mb",
            "system_cpu_percent",
          ].map((k) => (
            <div key={k}>
              <dt>{k.replaceAll("_", " ")}</dt>
              <dd>{display(metrics[k])}</dd>
            </div>
          ))}
        </dl>
      </Panel>
    </>
  );
}
export function SOSPanel() {
  const sos = useApp((s) => s.sos);
  const [confirm, setConfirm] = useState(false);
  return (
    <Panel title="Emergency SOS">
      <p>
        Sends an emergency email through the device’s configured notification
        service.
      </p>
      <Badge>{sos?.status || "Unknown"}</Badge>
      {!!sos?.cooldown_remaining_sec && (
        <p>Cooldown: {Math.ceil(sos.cooldown_remaining_sec)} seconds</p>
      )}
      {!confirm ? (
        <button className="danger" onClick={() => setConfirm(true)}>
          Emergency SOS
        </button>
      ) : (
        <div
          className="confirmation"
          role="group"
          aria-label="Confirm emergency alert"
        >
          <p>Send an emergency alert to the configured caregiver now?</p>
          <Action
            danger
            label="Confirm and send SOS"
            run={async () => {
              const r = await api.action("/api/sos");
              void api
                .sos()
                .then((sos) => useApp.getState().set({ sos }))
                .catch(() => {});
              return r;
            }}
          />
          <button onClick={() => setConfirm(false)}>Cancel</button>
        </div>
      )}
      {sos?.last_result && (
        <div className="result">
          <h3>
            {sos.last_result.success
              ? "Emergency alert sent"
              : "Latest SOS attempt"}
          </h3>
          <p>
            {sos.last_result.status} ·{" "}
            {new Date(sos.last_result.event_time * 1000).toLocaleString()}
          </p>
          <p>
            Location:{" "}
            {sos.last_result.location.lat == null
              ? "Unavailable"
              : `${sos.last_result.location.lat}, ${sos.last_result.location.lon}`}
          </p>
        </div>
      )}
    </Panel>
  );
}
