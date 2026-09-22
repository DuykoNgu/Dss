import assert from "node:assert/strict";
import test from "node:test";
import { latestClosedSessionDate } from "../src/common/marketSession.js";

test("latest closed session respects Vietnam close time and weekends", () => {
  assert.deepEqual(
    [
      "2026-09-21T08:59:00+07:00",
      "2026-09-21T15:00:00+07:00",
      "2026-09-20T18:00:00+07:00",
    ].map(latestClosedSessionDate),
    ["2026-09-18", "2026-09-21", "2026-09-18"],
  );
});
