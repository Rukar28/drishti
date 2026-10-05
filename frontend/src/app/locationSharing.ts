import { api } from "./api";
import { useApp } from "./store";

let watch: number | null = null;
let generation = 0;

export function stopLocationSharing() {
  generation++;
  if (watch !== null) navigator.geolocation.clearWatch(watch);
  watch = null;
  useApp.getState().set({ locationSharing: false });
}

export function startLocationSharing() {
  if (watch !== null) return;
  if (!navigator.geolocation) {
    useApp
      .getState()
      .set({ locationError: "Browser geolocation unavailable." });
    return;
  }
  const current = ++generation;
  let pending = false;
  const fail = (message: string) => {
    if (current !== generation) return;
    stopLocationSharing();
    useApp.getState().set({ locationError: message });
  };
  useApp.getState().set({ locationSharing: true, locationError: "" });
  watch = navigator.geolocation.watchPosition(
    (position) => {
      if (current !== generation || pending) return;
      pending = true;
      void api
        .locationUpdate(position.coords, position.timestamp)
        .then((location) => {
          if (current === generation) useApp.getState().set({ location });
        })
        .catch((error: Error) => fail(error.message))
        .finally(() => {
          pending = false;
        });
    },
    (error) => fail(error.message),
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 5000 },
  );
}
