import { useApp } from "../store";
import { Panel } from "../components";
import History from "./History";
export default function Overview() {
  const s = useApp();
  return (
    <>
      <h1>Caregiver overview</h1>
      <p className="muted">Live device information · Trusted local prototype</p>
      {s.sos?.last_result && (
        <div className="offline">
          SOS: {s.sos.last_result.status} ·{" "}
          {new Date(s.sos.last_result.event_time * 1000).toLocaleString()}
        </div>
      )}
      <div className="summary-grid">
        {[
          ["Device", s.online ? "Online" : "Offline"],
          ["Mode", s.world?.mode],
          ["Location", s.location?.status],
          ["Navigation", s.navigation?.state],
          ["Voice", s.world?.voice_state],
          ["SOS", s.sos?.status],
        ].map(([k, v]) => (
          <Panel key={k} title={k || ""}>
            <strong>{v || "Unavailable"}</strong>
          </Panel>
        ))}
      </div>
      <History />
    </>
  );
}
