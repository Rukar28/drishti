import { useState } from "react";
import { api } from "../api";
import { useApp } from "../store";
import { Panel, Result } from "../components";
import VoiceStatus from "../VoiceStatus";
import type { ActionResult } from "../types";
const features = [
  ["find", "Find an object", "Locate a target in the camera view."],
  ["read", "Read text", "Read visible text aloud."],
  ["ask", "Ask about the scene", "Ask the device’s visual assistant."],
  ["color", "Identify color", "Check an object’s color."],
  ["currency", "Recognize currency", "Read visible currency details."],
  [
    "medicine",
    "Read a medicine label",
    "Read the label and backend-provided information.",
  ],
];
export default function Assist() {
  const [feature, setFeature] = useState("find"),
    [input, setInput] = useState(""),
    [full, setFull] = useState(false),
    [busy, setBusy] = useState(false),
    [result, setResult] = useState<ActionResult | null>(null),
    [error, setError] = useState("");
  const world = useApp((s) => s.world),
    status = useApp((s) => s.status);
  const needsInput = feature === "find" || feature === "ask";
  const matched = world?.objects.filter((o) =>
    o.label
      .toLowerCase()
      .includes((status?.target_find_query || "").toLowerCase()),
  );
  return (
    <>
      <h1>Assist</h1>
      <VoiceStatus />
      <p className="muted">
        Manual fallback controls. These use the same device camera and backend
        features.
      </p>
      <div className="assist-layout">
        <div className="feature-list">
          {features.map(([id, title, description]) => (
            <button
              disabled={busy}
              key={id}
              className={feature === id ? "selected" : ""}
              onClick={() => {
                setFeature(id);
                setInput("");
                setResult(null);
                setError("");
              }}
            >
              <strong>{title}</strong>
              <small>{description}</small>
            </button>
          ))}
        </div>
        <Panel title={features.find((f) => f[0] === feature)![1]}>
          <form
            onSubmit={async (e) => {
              e.preventDefault();
              setBusy(true);
              setResult(null);
              setError("");
              try {
                const body =
                  feature === "find"
                    ? { target: input.trim() }
                    : feature === "ask"
                      ? { query: input.trim() }
                      : feature === "read"
                        ? { full_reading: full }
                        : feature === "color"
                          ? { target: input.trim() || null }
                          : {};
                setResult(await api.action(`/api/v1/${feature}`, body));
              } catch (e) {
                setError((e as Error).message);
              } finally {
                setBusy(false);
              }
            }}
          >
            {(needsInput || feature === "color") && (
              <label>
                {feature === "ask" ? "Your question" : "Target object"}
                <input
                  required={needsInput}
                  maxLength={500}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={
                    feature === "ask"
                      ? "What is in front of me?"
                      : feature === "color"
                        ? "Optional: bottle, shirt…"
                        : "For example, bottle"
                  }
                />
              </label>
            )}
            {feature === "read" && (
              <label className="check">
                <input
                  type="checkbox"
                  checked={full}
                  onChange={(e) => setFull(e.target.checked)}
                />
                Read full text
              </label>
            )}
            <button
              className="primary"
              disabled={busy || (needsInput && !input.trim())}
            >
              {busy ? "Processing…" : `Start ${feature}`}
            </button>
          </form>
          {error && (
            <p role="alert" className="error">
              {error}
            </p>
          )}
          <Result result={result} />
          {feature === "find" && status?.target_find_query && (
            <div className="result">
              <strong>Target: {status.target_find_query}</strong>
              <p>
                {matched?.length
                  ? "Tracking"
                  : "Searching — target not detected yet."}
              </p>
              {matched?.map((o) => (
                <p key={o.id}>
                  {o.label}: {o.position} ·{" "}
                  {o.distance_m == null
                    ? "Distance unavailable"
                    : `${o.distance_m} m`}
                </p>
              ))}
            </div>
          )}
          <p className="muted">
            Results come from the backend. Keep the target within the camera’s
            view.
          </p>
        </Panel>
      </div>
      <p className="muted">
        Focus mode and face recognition are unavailable in this backend.
      </p>
    </>
  );
}
