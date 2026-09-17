async function getJson(path, options) {
  const response = await fetch(path, options);
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error("Máy chủ trả về dữ liệu không hợp lệ.");
  }
  if (!response.ok) {
    throw new Error(body.error || "Không tải được dữ liệu.");
  }
  return body;
}

export function getMarket(signal) {
  return getJson("/api/market", { signal }).then((market) => {
    if (
      market.contract_version !== 1 ||
      !Array.isArray(market.stocks) ||
      typeof market.data_as_of !== "string" ||
      typeof market.snapshot_generated_at !== "string" ||
      typeof market.model_id !== "string" ||
      typeof market.model_version !== "string" ||
      typeof market.model_spec?.label_strategy !== "string" ||
      !Number.isInteger(market.model_spec?.horizon) ||
      !["ok", "error"].includes(market.sync_status) ||
      !["live", "degraded", "unavailable", "closed"].includes(
        market.quote_status,
      )
    ) {
      throw new Error("Dữ liệu thị trường không đúng phiên bản API.");
    }
    return market;
  });
}

export function getHistory(symbol, signal) {
  return getJson(`/api/history?symbol=${encodeURIComponent(symbol)}`, {
    signal,
  });
}

export function getRecommendation(symbol, signal) {
  return getJson(`/api/recommend?symbol=${encodeURIComponent(symbol)}`, {
    signal,
  });
}

export function getProfitLoss(position) {
  return getJson("/api/profit-loss", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(position),
  });
}

export function askAssistant(question) {
  return getJson("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}
