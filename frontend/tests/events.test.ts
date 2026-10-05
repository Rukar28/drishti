import { test } from "node:test";
import assert from "node:assert/strict";
import { parseEnvelope } from "../src/app/events.ts";
const envelope = (type: string, data: unknown) =>
  JSON.stringify({ type, timestamp: "2026-10-05T10:00:00Z", data });
test("ignores truncated JSON and non-object envelopes", () => {
  for (const value of ["{", "null", "[]", '"text"', null, 42])
    assert.equal(parseEnvelope(value), null);
});
test("rejects invalid data and timestamps", () => {
  assert.equal(parseEnvelope(envelope("SOS_ALERT", [])), null);
  assert.equal(
    parseEnvelope('{"type":"ERROR","timestamp":"invalid","data":{}}'),
    null,
  );
});
test("rejects malformed world objects before rendering", () => {
  assert.equal(
    parseEnvelope(
      envelope("WORLD_UPDATE", {
        mode: "GUIDANCE",
        voice_state: "IDLE",
        hazards: [],
        objects: [null],
      }),
    ),
    null,
  );
});
test("accepts actual world contract with unknown physical distance", () => {
  const data = {
    mode: "GUIDANCE",
    voice_state: "IDLE",
    hazards: [],
    objects: [
      { id: "bottle_1", label: "bottle", confidence: 0.9, distance_m: null },
    ],
  };
  assert.deepEqual(parseEnvelope(envelope("WORLD_UPDATE", data))?.data, data);
});
test("preserves SOS failure and navigation progress without invented success", () => {
  assert.equal(
    parseEnvelope(envelope("SOS_ALERT", { status: "not_configured" }))?.data
      .status,
    "not_configured",
  );
  assert.deepEqual(
    parseEnvelope(
      envelope("NAVIGATION_UPDATE", {
        status: "GUIDING",
        navigation: { progress: 0.4 },
      }),
    )?.data.navigation,
    { progress: 0.4 },
  );
});
