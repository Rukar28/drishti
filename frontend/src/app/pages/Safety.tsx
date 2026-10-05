import { useApp } from "../store";
import { DeviceStatus, Panel, SOSPanel, display } from "../components";
export default function Safety() {
  const hazards = useApp((s) => s.world?.hazards);
  return (
    <>
      <h1>Safety</h1>
      <div className="grid">
        <SOSPanel />
        <Panel title="Current hazards">
          {hazards?.length ? (
            hazards.map((h, i) => (
              <p className="error" key={i}>
                {display(h)}
              </p>
            ))
          ) : (
            <p>
              {hazards ? "No hazards reported." : "Hazard state unavailable."}
            </p>
          )}
        </Panel>
      </div>
      <DeviceStatus />
    </>
  );
}
