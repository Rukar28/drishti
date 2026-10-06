import { useApp } from "./store";
import { Panel, Result } from "./components";

const labels: Record<string, string> = {
  LISTENING_FOR_WAKE: "Listening for “Hey Mycroft”",
  WAKE_DETECTED: "Hey Mycroft detected",
  LISTENING_FOR_COMMAND: "Listening for your command",
  PROCESSING: "Transcribing your command",
  EXECUTING: "Executing your command",
  SPEAKING: "Speaking",
  IDLE: "Voice idle",
  COOLDOWN: "Preparing to listen",
};
export default function VoiceStatus() {
  const s = useApp();
  const wake = s.health?.subsystems.wake_word;
  const available = typeof wake === "object" && wake.available === true;
  const real = typeof wake === "object" && wake.is_real_wake_word === true;
  const wakeError =
    typeof wake === "object" && typeof wake.last_error === "string"
      ? wake.last_error
      : "";
  const state = s.world?.voice_state || s.status?.voice_state || "";
  const connected = s.online && s.connection === "Connected";
  return (
    <section className="voice-hud" aria-label="Voice assistant status">
      <Panel title="Speak to VisionMate">
        <p>
          Say “Hey Mycroft”, then say your command when listening begins. For
          example, “What color is this?”
        </p>
        <p className="voice-current" role="status" aria-live="polite">
          {!connected
            ? "Voice status unavailable — reconnecting"
            : !available
              ? "Wake-word service unavailable"
              : !real
                ? "Development wake mode — real wake detection inactive"
                : s.speaking
                  ? "Speaking"
                  : labels[state] || "Waiting for backend voice state"}
        </p>
        {s.wakeDetectedAt && (
          <p>
            Wake detected at {new Date(s.wakeDetectedAt).toLocaleTimeString()}
          </p>
        )}
        <dl>
          <div>
            <dt>Command heard</dt>
            <dd>{s.voiceTranscript || "Waiting for a spoken command"}</dd>
          </div>
          <div>
            <dt>Recognized feature</dt>
            <dd>{s.voiceIntent || "—"}</dd>
          </div>
        </dl>
        {s.voiceResult && (
          <>
            <h3>Last command result</h3>
            <Result result={s.voiceResult} />
          </>
        )}
        {s.voiceError && (
          <p className="error" role="alert">
            {s.voiceError}
          </p>
        )}
        {wakeError && (
          <p className="error" role="alert">
            {wakeError}
          </p>
        )}
        <details>
          <summary>Recent voice events</summary>
          <ol>
            {s.events
              .filter(
                (e) => e.type.startsWith("VOICE_") || e.type.startsWith("TTS_"),
              )
              .slice(0, 6)
              .map((e, i) => (
                <li key={`${e.timestamp}-${i}`}>
                  {new Date(e.timestamp).toLocaleTimeString()} ·{" "}
                  {e.type.replaceAll("_", " ")}
                  {typeof e.data.intent === "string"
                    ? ` · ${e.data.intent}`
                    : typeof e.data.state === "string"
                      ? ` · ${e.data.state.replaceAll("_", " ")}`
                      : ""}
                </li>
              ))}
          </ol>
        </details>
      </Panel>
    </section>
  );
}
