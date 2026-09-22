import assert from "node:assert/strict";
import test from "node:test";
import {
  askAssistant,
  getHistory,
  getMarket,
  getProfitLoss,
  getRecommendation,
} from "../src/common/api.js";

test("frontend calls the backend routes with the expected payload", async () => {
  const calls = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (path, options) => {
    calls.push({ path, options });
    return {
      ok: true,
      json: async () =>
        path.startsWith("/api/market?")
          ? {
              contract_version: 1,
              stocks: [],
              data_as_of: "2026-09-17",
              snapshot_generated_at: "2026-09-17T15:30:00+07:00",
              model_id: "WEB_POOLED_EXCESS_H20_HISTORY",
              model_version: "abc123",
              model_spec: { label_strategy: "excess", horizon: 20 },
              quote_status: "closed",
              sync_status: "ok",
            }
          : { ok: true },
    };
  };

  try {
    await getMarket("2026-09-18");
    await getHistory("FPT");
    await getRecommendation("HPG");
    await getProfitLoss({
      symbol: "ACB",
      entry_date: "2026-09-15",
      buy_price: 20,
      quantity: 100,
    });
    await askAssistant("So sánh FPT và HPG");
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.deepEqual(
    calls.map(({ path }) => path),
    [
      "/api/market?session_date=2026-09-18",
      "/api/history?symbol=FPT",
      "/api/recommend?symbol=HPG",
      "/api/profit-loss",
      "/api/ask",
    ],
  );
  assert.equal(calls[3].options.method, "POST");
  assert.deepEqual(JSON.parse(calls[3].options.body), {
    symbol: "ACB",
    entry_date: "2026-09-15",
    buy_price: 20,
    quantity: 100,
  });
  assert.deepEqual(JSON.parse(calls[4].options.body), {
    question: "So sánh FPT và HPG",
  });
});

test("market rejects incompatible API response", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ stocks: [] }),
  });
  try {
    await assert.rejects(
      getMarket("2026-09-18"),
      /không đúng phiên bản API/,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("backend errors reach the interface", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: false,
    json: async () => ({ error: "Mã không thuộc VN30" }),
  });

  try {
    await assert.rejects(getRecommendation("XYZ"), /Mã không thuộc VN30/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
