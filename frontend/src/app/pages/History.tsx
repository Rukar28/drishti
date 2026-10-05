import { useApp } from "../store";
import { Panel, display } from "../components";
export default function History() {
  const events = useApp((s) => s.events);
  return (
    <Panel title="Current-session history">
      <p className="muted">
        Received events only. History resets when this page is reloaded.
      </p>
      {!events.length && <p>No events received yet.</p>}
      <ol className="timeline">
        {events.map((e, i) => (
          <li key={`${e.timestamp}-${i}`}>
            <time>{new Date(e.timestamp).toLocaleTimeString()}</time>
            <div>
              <strong>{e.type.replaceAll("_", " ")}</strong>
              <p>
                {display(
                  e.data.message ||
                    e.data.text ||
                    e.data.transcript ||
                    e.data.status ||
                    e.data.mode ||
                    e.data.voice_state ||
                    e.type,
                )}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </Panel>
  );
}
