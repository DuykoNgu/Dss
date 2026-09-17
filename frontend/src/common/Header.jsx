import { Activity, ArrowRight, MessageCircle } from "lucide-react";
import { dateText } from "./format";

export default function Header({ market, onAsk }) {
  const quote = market?.stocks?.find((stock) => stock.quote)?.quote;
  const status = market?.quote_status;
  const label =
    status === "live"
      ? "GIÁ TRONG PHIÊN"
      : status === "degraded"
        ? "GIÁ TRONG PHIÊN CHƯA ĐỦ"
        : status === "unavailable"
          ? "CHƯA CÓ GIÁ TRONG PHIÊN"
          : "DỮ LIỆU CUỐI PHIÊN";
  return (
    <>
      <div className="statusline">
        <div className="container statusline-inner">
          <span
            className={`status-dot ${status === "unavailable" || status === "degraded" || market?.sync_status === "error" ? "warning" : ""}`}
          />{" "}
          {label} · ML {dateText(market?.data_as_of)}{" "}
          {quote && (
            <>
              <span className="status-separator">/</span> GIÁ LÚC{" "}
              {(quote.source_at || quote.updated_at).slice(11, 19)}{" "}
            </>
          )}
          <span className="status-right">
            VN30 · HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH
          </span>
        </div>
      </div>
      <header className="header">
        <div className="container header-inner">
          <a
            className="brand"
            href="#thi-truong"
            aria-label="DSS VN30, về bảng giá"
          >
            <span className="brand-mark">
              <Activity size={23} strokeWidth={2.6} />
            </span>
            <span className="brand-copy">
              <b>
                DSS<span>.</span>
              </b>
              <small>VN30 INTELLIGENCE</small>
            </span>
          </a>
          <nav className="navigation" aria-label="Điều hướng chính">
            <a href="#thi-truong" className="active">
              Thị trường
            </a>
            <a href="#kham-pha">Khám phá</a>
            <a href="#kiem-chung">Kiểm chứng</a>
          </nav>
          <button className="header-ask" type="button" onClick={onAsk}>
            <MessageCircle size={16} /> Hỏi trợ lý <ArrowRight size={15} />
          </button>
        </div>
      </header>
    </>
  );
}
