import { useState } from "react";
import { DeviceStatus, Panel } from "../components";
export default function Settings() {
  const [large, setLarge] = useState(
    localStorage.getItem("vm-large") === "true",
  );
  return (
    <>
      <h1>Settings</h1>
      <Panel title="Reading comfort">
        <label className="check">
          <input
            type="checkbox"
            checked={large}
            onChange={(e) => {
              setLarge(e.target.checked);
              localStorage.setItem("vm-large", String(e.target.checked));
              document.documentElement.classList.toggle(
                "large-text",
                e.target.checked,
              );
            }}
          />
          Larger text
        </label>
        <p className="muted">
          Motion follows your system’s reduced-motion preference.
        </p>
      </Panel>
      <DeviceStatus />
    </>
  );
}
