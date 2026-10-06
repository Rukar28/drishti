import { useState } from "react";
import { startLocationSharing, stopLocationSharing } from "../locationSharing";
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
  const [destination, setDestination] = useState("");
  const sharing = useApp((s) => s.locationSharing);
  const error = useApp((s) => s.locationError);
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
                onClick={sharing ? stopLocationSharing : startLocationSharing}
              >
                {!sharing
                  ? "Share this device’s location"
                  : "Stop sharing location"}
              </button>
              <p className="muted">
                Requires browser permission and GPS_SOURCE=browser on the
                backend. Sharing continues across pages until you stop it,
                switch roles, or close this tab.
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
              ["Next maneuver", nav?.next_instruction || nav?.next_maneuver],
              [
                "Distance to step end",
                nav?.distance_to_maneuver_m == null
                  ? null
                  : `${Math.round(nav.distance_to_maneuver_m)} m`,
              ],
              [
                "Remaining distance (estimate)",
                nav?.remaining_distance_m == null
                  ? null
                  : `${Math.round(nav.remaining_distance_m)} m`,
              ],
              [
                "Remaining duration (estimate)",
                nav?.remaining_duration_s == null
                  ? null
                  : `${Math.ceil(nav.remaining_duration_s / 60)} min`,
              ],
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
          {nav?.progress != null && (
            <label>
              Route progress{" "}
              <progress
                aria-label="Route progress"
                max={1}
                value={nav.progress}
              />
              <span>{Math.round(nav.progress * 100)}%</span>
            </label>
          )}
          {nav?.state === "PAUSED_GPS" && (
            <p role="alert" className="error">
              Location unavailable. Navigation guidance is paused until a fresh
              fix arrives.
            </p>
          )}
          <p className="muted">
            Remaining estimates use GPS projection onto the returned route. They
            are not live traffic estimates. Distance and ETA are the provider’s
            route estimate at calculation time.
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
