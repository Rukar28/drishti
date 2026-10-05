import type {
  ActionResult,
  Health,
  WorldState,
  Status,
  LocationState,
  NavigationState,
  SOSState,
} from "./types";
export const base = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
export const url = (path: string) => `${base}${path}`;
export const wsURL = (path: string) => {
  const u = new URL(url(path), window.location.href);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  return u.href;
};
export async function request<T>(path: string, body?: unknown): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(
    () => controller.abort(),
    body === undefined ? 15000 : 90000,
  );
  try {
    const response = await fetch(url(path), {
      method: body === undefined ? "GET" : "POST",
      signal: controller.signal,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : `Request failed (${response.status})`,
      );
    }
    return await response.json();
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError")
      throw new Error(
        "Backend request timed out. The operation may still be running; use STOP to interrupt.",
      );
    throw e;
  } finally {
    clearTimeout(timer);
  }
}
export const api = {
  health: () => request<Health>("/health"),
  status: () => request<Status>("/api/status"),
  world: () => request<WorldState>("/api/world"),
  metrics: () => request<Record<string, number | string | boolean>>("/metrics"),
  location: () => request<LocationState>("/api/location"),
  navigation: () => request<NavigationState>("/api/navigation"),
  sos: () => request<SOSState>("/api/sos"),
  action: (path: string, body = {}) => request<ActionResult>(path, body),
  locationUpdate: (coords: GeolocationCoordinates) =>
    request<LocationState>("/api/location", {
      lat: coords.latitude,
      lon: coords.longitude,
      accuracy: coords.accuracy,
    }),
};
