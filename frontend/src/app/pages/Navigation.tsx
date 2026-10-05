import { useEffect, useState } from "react";
import { Navigate, Route } from "react-router-dom";
import { api } from "../api";
import { useApp } from "../store";
import { Action, Badge, MapView, Panel } from "../components";
export default function Navigation({
  caregiver = false,
}: {
  caregiver?: boolean;
}) {
  const nav = useApp((s) => s.navigation),
    loc = useApp((s) => s.location);
  const [destination, setDestination] = useState(""),
    [watch, setWatch] = useState<number | null>(null),
    [error, setError] = useState("");
  useEffect(
    () => () => {
      if (watch !== null) navigator.geolocation.clearWatch(watch);
    },
    [watch],
  );
  return (
    <>
      <h1>{caregiver ? "Device location" : "Navigate"}</h1>
      <p className="muted">
        {nav?.provider === "mock"
          ? "Demo Navigation — offline straight-line directions."
          : "Walking directions from the configured backend provider."}
      </p>
      <div className="grid">
        <Panel title="Location">
          <Badge>{loc?.status || "Unavailable"}</Badge>
          <MapView />
          {!caregiver && (
            <>
              <button
                onClick={() => {
                  if (watch !== null) {
                    navigator.geolocation.clearWatch(watch);
                    setWatch(null);
                    return;
                  }
                  if (!navigator.geolocation) {
                    setError("Browser geolocation unavailable.");
                    return;
                  }
                  setError("");
                  const id = navigator.geolocation.watchPosition(
                    (p) => {
                      void api
                        .locationUpdate(p.coords)
                        .then((location) => useApp.getState().set({ location }))
                        .catch((e) => {
                          setError(e.message);
                          navigator.geolocation.clearWatch(id);
                          setWatch(null);
                        });
                    },
                    (e) => {
                      setError(e.message);
                      navigator.geolocation.clearWatch(id);
                      setWatch(null);
                    },
                    {
                      enableHighAccuracy: true,
                      timeout: 15000,
                      maximumAge: 5000,
                    },
                  );
                  setWatch(id);
                }}
              >
                {watch === null
                  ? "Share this device’s location"
                  : "Stop sharing location"}
              </button>
              <p className="muted">
                Requires browser permission and GPS_SOURCE=browser on the
                backend. Sharing ends when you leave this page.
              </p>
            </>
          )}
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </Panel>
        <Panel title="Route">
          {!caregiver && (
            <>
              <label>
                Destination
                <input
                  value={destination}
                  onChange={(e) => setDestination(e.target.value)}
                  placeholder="Place or address"
                  maxLength={300}
                />
              </label>
              <Action
                label="Start navigation"
                run={() =>
                  destination.trim()
                    ? api.action("/api/navigation", {
                        destination: destination.trim(),
                      })
                    : Promise.reject(new Error("Enter a destination."))
                }
              />
              <Action
                label="Stop navigation"
                run={() => api.action("/api/navigation/stop")}
              />
            </>
          )}
          <dl>
            {[
              ["Provider", nav?.provider],
              ["State", nav?.state],
              ["Destination", nav?.active_destination],
              ["Instruction", nav?.instruction],
              [
                "Route distance",
                nav?.route?.distance_m == null
                  ? null
                  : `${nav.route.distance_m} m`,
              ],
              [
                "Route ETA",
                nav?.route?.duration_s == null
                  ? null
                  : `${Math.ceil(nav.route.duration_s / 60)} min`,
              ],
            ].map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{v || "Unavailable"}</dd>
              </div>
            ))}
          </dl>
          <p className="muted">
            Distance and ETA are the provider’s route estimate at calculation
            time.
          </p>
          {nav?.error && <p className="error">{nav.error}</p>}
          {nav?.route?.warnings?.map((w) => (
            <p key={w}>{w}</p>
          ))}
          {nav?.route?.steps && (
            <ol>
              {nav.route.steps.map((s, i) => (
                <li
                  key={i}
                  aria-current={i === nav.step_index ? "step" : undefined}
                >
                  {s.instruction}
                  {s.distance_m != null ? ` · ${s.distance_m} m` : ""}
                </li>
              ))}
            </ol>
          )}
        </Panel>
      </div>
    </>
  );
}
