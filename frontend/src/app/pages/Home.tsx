import { api } from "../api";
import { useApp } from "../store";
import { Action, Badge, LiveFeed, Panel, display } from "../components";
export default function Home() {
  const assistant = useApp((s) => s.assistant);
  const world = useApp((s) => s.world),
    status = useApp((s) => s.status);
  return (
    <>
      <div className="page-heading">
        <div>
          <p className="eyebrow">YOUR EVERYDAY COMPANION</p>
          <h1>Live vision</h1>
          <p className="muted">
            Your surroundings, with a little more clarity.
          </p>
        </div>
        <Badge>{world?.mode || "Mode unavailable"}</Badge>
      </div>
      <div className="live-grid">
        <Panel title="Camera">
          <LiveFeed />
        </Panel>
        <div className="stack">
          <Panel title="Assistant">
            <p className="narrative" aria-live="polite">
              {assistant ||
                status?.last_spoken_narrative ||
                "No spoken guidance yet."}
            </p>
            <dl>
              <div>
                <dt>Voice</dt>
                <dd>{world?.voice_state || "Unavailable"}</dd>
              </div>
              <div>
                <dt>Guidance mode</dt>
                <dd>{world?.mode || "Unavailable"}</dd>
              </div>
            </dl>
            <Action
              label="Enable guidance"
              run={() => api.action("/api/v1/mode", { mode: "guidance" })}
            />
          </Panel>
          <Panel title="Safety nearby">
            {world?.hazards.length ? (
              world.hazards.map((h, i) => (
                <p className="error" key={i}>
                  {display(h)}
                </p>
              ))
            ) : (
              <p className="muted">
                {world
                  ? "No hazards currently reported."
                  : "Safety state unavailable."}
              </p>
            )}
          </Panel>
        </div>
      </div>
      <Panel title="Detected objects">
        <div className="objects">
          {world?.objects.map((o) => (
            <article key={o.id}>
              <h3>{o.label}</h3>
              <p>
                {o.position} ·{" "}
                {o.distance_m == null
                  ? "Distance unavailable"
                  : `${o.distance_m.toFixed(1)} m`}
              </p>
              <small>
                {o.velocity} · {Math.round(o.confidence * 100)}% confidence
              </small>
            </article>
          ))}
        </div>
        {!world?.objects.length && (
          <p className="muted">No active objects reported.</p>
        )}
      </Panel>
    </>
  );
}
