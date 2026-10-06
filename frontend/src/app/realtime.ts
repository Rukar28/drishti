import { api, wsURL } from "./api";
import { useApp } from "./store";
import { parseEnvelope } from "./events";
import type { WorldState, NavigationState, LocationState } from "./types";
export function connectSocket(
  path: string,
  message: (event: MessageEvent) => void,
  state: (value: string) => void,
) {
  let socket: WebSocket | null = null,
    timer = 0,
    stopped = false,
    attempt = 0;
  function connect() {
    if (stopped) return;
    state(attempt ? "Reconnecting" : "Connecting");
    socket = new WebSocket(wsURL(path));
    socket.binaryType = "blob";
    socket.onopen = () => {
      attempt = 0;
      state("Connected");
    };
    socket.onmessage = message;
    socket.onerror = () => socket?.close();
    socket.onclose = () => {
      if (stopped) return;
      state("Reconnecting");
      timer = window.setTimeout(
        connect,
        Math.min(15000, 1000 * 2 ** attempt++),
      );
    };
  }
  connect();
  return () => {
    stopped = true;
    clearTimeout(timer);
    if (socket) {
      socket.onclose = null;
      socket.onmessage = null;
      socket.close();
    }
  };
}
export function startRealtime() {
  let stopped = false,
    polling = false;
  async function poll() {
    if (polling || stopped) return;
    polling = true;
    const results = await Promise.allSettled([
      api.health(),
      api.status(),
      api.world(),
      api.location(),
      api.navigation(),
      api.sos(),
      api.metrics(),
    ]);
    if (!stopped) {
      const keys = [
        "health",
        "status",
        "world",
        "location",
        "navigation",
        "sos",
        "metrics",
      ];
      const values: Record<string, unknown> = {};
      results.forEach((r, i) => {
        if (r.status === "fulfilled") values[keys[i]] = r.value;
      });
      const online = results[0].status === "fulfilled";
      if (results[0].status === "fulfilled") {
        const speech = results[0].value.subsystems.speech_engine;
        if (typeof speech === "object" && typeof speech.speaking === "boolean")
          values.speaking = speech.speaking;
      }
      useApp.getState().set({
        ...values,
        online,
        updated: online ? Date.now() : useApp.getState().updated,
        error: online
          ? results.some((r) => r.status === "rejected")
            ? "Some device information is unavailable. Retrying automatically."
            : ""
          : "Backend disconnected. Retrying automatically.",
      });
    }
    polling = false;
  }
  const close = connectSocket(
    "/ws/events",
    (message) => {
      try {
        const e = parseEnvelope(message.data);
        if (!e) return;
        const store = useApp.getState();
        if (e.type === "VOICE_STARTED")
          store.set({
            wakeDetectedAt: e.timestamp,
            voiceTranscript: "",
            voiceIntent: "",
            voiceResult: null,
            voiceError: "",
          });
        if (
          e.type === "VOICE_TRANSCRIPT" &&
          typeof e.data.transcript === "string"
        )
          store.set({ voiceTranscript: e.data.transcript });
        if (e.type === "VOICE_EXECUTING" && typeof e.data.intent === "string")
          store.set({ voiceIntent: e.data.intent, voiceResult: null });
        if (e.type === "VOICE_COMMAND") store.set({ voiceResult: e.data });
        if (e.type === "TTS_STARTED") store.set({ speaking: true });
        if (e.type === "TTS_FINISHED") store.set({ speaking: false });
        if (
          e.type === "ERROR" &&
          ["voice", "tts"].includes(String(e.data.source))
        )
          store.set({
            voiceError: String(e.data.message || "Voice service error"),
            speaking: false,
          });
        if (
          e.type === "NAVIGATION_UPDATE" &&
          e.data.navigation &&
          typeof e.data.navigation === "object"
        )
          store.set({ navigation: e.data.navigation as NavigationState });
        if (
          e.type === "SENSOR_UPDATE" &&
          e.data.location &&
          typeof e.data.location === "object"
        )
          store.set({ location: e.data.location as LocationState });
        if (e.type === "TTS_STARTED" && typeof e.data.text === "string")
          store.set({ assistant: e.data.text });
        if (
          e.type === "WORLD_UPDATE" &&
          Array.isArray(e.data.objects) &&
          Array.isArray(e.data.hazards)
        )
          store.set({ world: e.data as unknown as WorldState });
        else if (
          ["VOICE_STATE", "SYSTEM_STATUS"].includes(e.type) &&
          store.world
        )
          store.set({
            world: {
              ...store.world,
              voice_state: String(
                e.data.state || e.data.voice_state || store.world.voice_state,
              ),
            },
          });
        if (
          [
            "SOS_ALERT",
            "MODE_CHANGED",
            "ERROR",
            "HAZARD",
            "VOICE_COMMAND",
            "VOICE_TRANSCRIPT",
            "VOICE_STARTED",
            "VOICE_EXECUTING",
            "VOICE_STATE",
            "TTS_STARTED",
            "TTS_FINISHED",
            "SYSTEM_STATUS",
          ].includes(e.type)
        )
          store.event(e);
        if (e.type === "NAVIGATION_UPDATE" && e.data.status !== "GUIDING")
          store.event(e);
        if (["SOS_ALERT", "MODE_CHANGED"].includes(e.type)) void poll();
      } catch {
        /* Ignore malformed envelopes; next snapshot repairs state. */
      }
    },
    (connection) => useApp.getState().set({ connection }),
  );
  void poll();
  const timer = window.setInterval(poll, 5000);
  return () => {
    stopped = true;
    close();
    clearInterval(timer);
  };
}
