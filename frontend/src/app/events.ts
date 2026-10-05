/** Boundary validation shared by the single event WebSocket consumer. */
import type { Envelope } from "./types";
export function parseEnvelope(raw: unknown): Envelope | null {
  if (typeof raw !== "string") return null;
  try {
    const e = JSON.parse(raw);
    if (
      !e ||
      typeof e.type !== "string" ||
      typeof e.timestamp !== "string" ||
      !Number.isFinite(Date.parse(e.timestamp)) ||
      !e.data ||
      typeof e.data !== "object" ||
      Array.isArray(e.data)
    )
      return null;
    if (e.type === "WORLD_UPDATE") {
      if (
        typeof e.data.mode !== "string" ||
        typeof e.data.voice_state !== "string" ||
        !Array.isArray(e.data.hazards) ||
        !Array.isArray(e.data.objects)
      )
        return null;
      if (
        e.data.objects.some(
          (o: unknown) =>
            !o ||
            typeof o !== "object" ||
            typeof (o as Record<string, unknown>).label !== "string" ||
            typeof (o as Record<string, unknown>).id !== "string" ||
            typeof (o as Record<string, unknown>).confidence !== "number" ||
            ((o as Record<string, unknown>).distance_m !== null &&
              typeof (o as Record<string, unknown>).distance_m !== "number"),
        )
      )
        return null;
    }
    return e;
  } catch {
    return null;
  }
}
