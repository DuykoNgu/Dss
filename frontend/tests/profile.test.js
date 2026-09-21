import assert from "node:assert/strict";
import test from "node:test";
import { profileSummary } from "../src/features/onboarding/profile.js";

test("onboarding summary adapts to investor experience and goal", () => {
  assert.match(
    profileSummary({ experience: "new", knowledge: "starter", goal: "learn" }),
    /giải thích từ nền tảng.*giải thích tín hiệu/,
  );
  assert.match(
    profileSummary({ experience: "experienced", knowledge: "advanced", goal: "verify" }),
    /đi thẳng vào dữ liệu.*rủi ro/,
  );
  assert.match(profileSummary({}), /hoàn tất ba lựa chọn/);
});
