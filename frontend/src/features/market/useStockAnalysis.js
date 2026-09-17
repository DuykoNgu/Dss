import { useEffect, useState } from "react";
import { getHistory, getRecommendation } from "../../common/api";

export default function useStockAnalysis(symbol, date) {
  const [history, setHistory] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!symbol) return;
    const controller = new AbortController();
    setHistory([]);
    setComparison(null);
    setError("");
    Promise.allSettled([
      getHistory(symbol, controller.signal),
      getRecommendation(symbol, controller.signal),
    ]).then(([prices, advice]) => {
      if (controller.signal.aborted) return;
      if (prices.status === "fulfilled") setHistory(prices.value.candles);
      if (advice.status === "fulfilled") setComparison(advice.value);
      const failed =
        prices.status === "rejected"
          ? prices
          : advice.status === "rejected"
            ? advice
            : null;
      if (failed) setError(failed.reason.message);
    });
    return () => controller.abort();
  }, [symbol, date]);
  return { history, comparison, error };
}
