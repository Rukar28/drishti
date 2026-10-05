import { create } from "zustand";
import type {
  Health,
  Status,
  WorldState,
  LocationState,
  NavigationState,
  SOSState,
  Envelope,
} from "./types";
interface State {
  assistant: string;
  online: boolean;
  connection: string;
  error: string;
  updated: number | null;
  world: WorldState | null;
  health: Health | null;
  status: Status | null;
  location: LocationState | null;
  navigation: NavigationState | null;
  sos: SOSState | null;
  metrics: Record<string, number | string | boolean>;
  events: Envelope[];
  set: (s: Partial<State>) => void;
  event: (e: Envelope) => void;
}
export const useApp = create<State>((set) => ({
  assistant: "",
  online: false,
  connection: "Connecting",
  error: "",
  updated: null,
  world: null,
  health: null,
  status: null,
  location: null,
  navigation: null,
  sos: null,
  metrics: {},
  events: [],
  set: (s) => set(s),
  event: (e) => set((s) => ({ events: [e, ...s.events].slice(0, 150) })),
}));
