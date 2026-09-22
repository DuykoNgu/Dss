import { useEffect, useState } from "react";
import { ArrowUpRight } from "lucide-react";
import { getMarket } from "./common/api";
import { latestClosedSessionDate } from "./common/marketSession";
import Header from "./common/Header";
import MarketBoard from "./features/market/MarketBoard";
import StockAnalysisDialog from "./features/market/StockAnalysisDialog";
import Discovery from "./features/discovery/Discovery";
import Research from "./features/research/Research";
import ChatWidget from "./features/assistant/ChatWidget";
import Onboarding from "./features/onboarding/Onboarding";

const PROFILE_KEY = "dss-investor-profile";

export default function App() {
  const [market, setMarket] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [error, setError] = useState("");
  const [chatOpen, setChatOpen] = useState(false);
  const [chatDraft, setChatDraft] = useState("");
  const [showOnboarding, setShowOnboarding] = useState(() => {
    try {
      return !localStorage.getItem(PROFILE_KEY);
    } catch {
      return true;
    }
  });
  useEffect(() => {
    const controller = new AbortController();
    let timer;
    let inFlight = false;
    async function refresh() {
      if (inFlight) return;
      inFlight = true;
      try {
        const result = await getMarket(
          latestClosedSessionDate(),
          controller.signal,
        );
        setMarket(result);
        setError("");
      } catch (reason) {
        if (reason.name !== "AbortError") setError(reason.message);
      } finally {
        inFlight = false;
        clearTimeout(timer);
        if (!controller.signal.aborted) timer = setTimeout(refresh, 20000);
      }
    }
    refresh();
    const onFocus = () => {
      clearTimeout(timer);
      if (!controller.signal.aborted) refresh();
    };
    window.addEventListener("focus", onFocus);
    return () => {
      controller.abort();
      clearTimeout(timer);
      window.removeEventListener("focus", onFocus);
    };
  }, []);
  const stocks = market?.stocks || [];
  const selected = stocks.find((stock) => stock.symbol === selectedSymbol);
  function openChat(question = "") {
    setSelectedSymbol("");
    setChatDraft(question);
    setChatOpen(true);
  }
  function choose(symbol) {
    setSelectedSymbol(symbol);
    setChatOpen(false);
  }
  function completeOnboarding(profile) {
    try {
      localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    } catch {
      // The dashboard still works when browser storage is unavailable.
    }
    setShowOnboarding(false);
  }
  if (showOnboarding) {
    return (
      <Onboarding
        onComplete={completeOnboarding}
        onSkip={() => completeOnboarding({ skipped: true })}
      />
    );
  }
  return (
    <>
      <Header market={market} onAsk={() => openChat()} />
      <main className="container">
        {error && (
          <div className="error-banner">Không tải được dữ liệu: {error}</div>
        )}
        {market?.sync_status === "error" && (
          <div className="error-banner">
            Đồng bộ cuối ngày chưa hoàn tất. Đang hiển thị dữ liệu đã xác nhận
            ngày {market.data_as_of}.
          </div>
        )}
        <div className="workspace">
          <MarketBoard
            model={market?.model_spec}
            modelVersion={market?.model_version}
            stocks={stocks}
            selectedSymbol={selectedSymbol}
            onSelect={choose}
          />
        </div>
        <Discovery stocks={stocks} onSelect={choose} onAsk={openChat} />
        <Research research={market?.research} />
      </main>
      <footer className="footer">
        <div className="container">
          <span className="footer-brand">
            DSS<span>.</span> <small>VN30 RESEARCH</small>
          </span>
          <span>
            Giá trong phiên khi có · Điểm ML theo nến đã chốt · Không thực hiện
            giao dịch
          </span>
          <a href="#thi-truong">
            Về đầu trang <ArrowUpRight size={14} />
          </a>
        </div>
      </footer>
      {selected && (
        <StockAnalysisDialog
          stock={selected}
          onClose={() => setSelectedSymbol("")}
          onSelect={choose}
          onAsk={openChat}
        />
      )}
      <ChatWidget
        open={chatOpen}
        setOpen={setChatOpen}
        draft={chatDraft}
        setDraft={setChatDraft}
        onSelect={choose}
      />
    </>
  );
}
