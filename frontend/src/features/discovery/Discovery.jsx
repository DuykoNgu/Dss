import { ArrowRight } from "lucide-react";
import Change from "../../common/Change";
import {
  format,
  priceText,
  shownChange,
  shownPrice,
} from "../../common/format";

export default function Discovery({ stocks, onSelect, onAsk }) {
  const leaders = [...stocks]
    .filter((stock) => stock.ml_score != null)
    .sort((a, b) => b.ml_score - a.ml_score)
    .slice(0, 4);
  return (
    <section className="discovery" id="kham-pha">
      <div className="section-title">
        <div>
          <span className="section-kicker">
            <span className="kicker-square" /> KHÁM PHÁ / ML RANKING
          </span>
          <h2>Chưa biết bắt đầu từ đâu?</h2>
          <p>
            Đây là các mã đứng đầu xếp hạng ML trong phiên dữ liệu hiện tại.
            Chọn một mã để xem bối cảnh đầy đủ.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onAsk("Mã nào được ML xếp cao nhất?")}
        >
          Hỏi trợ lý ML <ArrowRight size={17} />
        </button>
      </div>
      <div className="leaders-grid">
        {leaders.map((stock, index) => (
          <button
            type="button"
            className="leader-card"
            key={stock.symbol}
            onClick={() => onSelect(stock.symbol)}
          >
            <span className="leader-rank">
              #{String(index + 1).padStart(2, "0")} <span>ML RANK</span>
            </span>
            <span className="leader-main">
              <strong>{stock.symbol}</strong>
              <span>
                {format.format(stock.ml_score)} <small>/ 100</small>
              </span>
            </span>
            <span className="leader-foot">
              <span>Giá {priceText(shownPrice(stock))}</span>
              <Change value={shownChange(stock)} compact />
            </span>
            <span className="leader-arrow">
              <ArrowRight size={17} />
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
