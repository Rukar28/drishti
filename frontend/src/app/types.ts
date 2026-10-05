export interface ObjectState {
  id: string;
  label: string;
  position: string;
  distance_m: number | null;
  velocity: string;
  confidence: number;
}
export interface LocationState {
  lat: number | null;
  lon: number | null;
  accuracy: number | null;
  timestamp: number | null;
  status: string;
  source: string;
}
export interface RouteState {
  status?: string;
  distance_m?: number;
  duration_s?: number;
  polyline?: string;
  steps?: { instruction: string; distance_m?: number }[];
  warnings?: string[];
}
export interface NavigationState {
  provider: string;
  status: string;
  state: string;
  active_destination?: string;
  destination_coords?: number[];
  instruction?: string;
  last_update?: number;
  step_index?: number;
  location_available?: boolean;
  progress?: number | null;
  remaining_distance_m?: number | null;
  remaining_duration_s?: number | null;
  distance_to_maneuver_m?: number | null;
  maneuver?: string | null;
  next_instruction?: string | null;
  next_maneuver?: string | null;
  route?: RouteState;
  error?: string;
}
export interface WorldState {
  mode: string;
  objects: ObjectState[];
  hazards: unknown[];
  voice_state: string;
  camera_connected: boolean;
  timestamp: string;
  navigation: NavigationState;
}
export type Subsystem = Record<string, unknown>;
export interface Health {
  app: string;
  version: string;
  cuda_available: boolean;
  gpu_name: string;
  subsystems: Record<string, Subsystem | number>;
}
export interface Status {
  mode: string;
  camera_source: string;
  camera_connected: boolean;
  camera_connection_state: string;
  fps: number;
  last_latency_ms: number;
  target_find_query: string | null;
  last_spoken_narrative: string;
  voice_state: string;
}
export interface ActionResult {
  status?: string;
  success?: boolean;
  has_text?: boolean;
  has_result?: boolean;
  text?: string;
  message?: string;
  result?: ActionResult;
  target?: string;
  color?: string;
  confidence?: number;
  [key: string]: unknown;
}
export interface Envelope {
  type: string;
  timestamp: string;
  data: Record<string, unknown>;
}
export interface SOSState {
  configured: boolean;
  status: string;
  cooldown_remaining_sec: number;
  last_result:
    (ActionResult & { event_time: number; location: LocationState }) | null;
}
