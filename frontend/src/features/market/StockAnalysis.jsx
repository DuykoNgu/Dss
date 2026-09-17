import { lazy, Suspense } from "react";
import {
  Calculator,
  ChevronRight,
  CircleHelp,
  MessageCircle,
  Sparkles,
} from "lucide-react";
import Change from "../../common/Change";
import {
  dateText,
  format,
  priceText,
  shownChange,
  shownPrice,
  signedText,
  tone,
} from "../../common/format";
import useStockAnalysis from "./useStockAnalysis";
import ProfitLossEstimator from "./ProfitLossEstimator";

const PriceChart = lazy(() => import("./PriceChart"));

export default function StockAnalysis({ stock, onSelect, onAsk }) {
  const { history, comparison, error } = useStockAnalysis(
    stock.symbol,
    stock.date,
  );
  const signalGap = stock.explanation
    ? stock.explanation.above - stock.explanation.below
    : 0;
  const threshold = stock.explanation?.label_threshold_pct;
  return (
    <section
      className="analysis-card"
      id="phan-tich"
      aria-label="Phân tích mã đã chọn"
    >
      <div className="analysis-heading">
        <span className="section-kicker">
          <span className="kicker-square" /> PHÂN TÍCH MÃ
        </span>
        <span className="analysis-live">
          <span />{" "}
          {stock.quote
            ? `GIÁ ${(stock.quote.source_at || stock.quote.updated_at).slice(11, 19)}`
            : "GIÁ ĐÓNG CỬA"}
        </span>
      </div>
      <div className="analysis-grid">
        <div className="analysis-main">
          <div className="selected-stock">
            <div>
              <span className="selected-label">MÃ ĐANG XEM</span>
              <h2>{stock.symbol}</h2>
              <p>Đóng cửa {dateText(stock.date)}</p>
            </div>
            <div className="selected-actions">
              <button
                type="button"
                className="estimate-stock"
                onClick={() =>
                  document.getElementById("profit-loss")?.scrollIntoView({
                    behavior: "smooth",
                  })
                }
              >
                <Calculator size={16} /> Xem lời/lỗ
              </button>
              <button
                type="button"
                className="ask-stock"
                onClick={() => onAsk(`${stock.symbol} thế nào theo ML?`)}
                title={`Hỏi ML về ${stock.symbol}`}
                aria-label={`Hỏi ML về ${stock.symbol}`}
              >
                <MessageCircle size={16} />
              </button>
            </div>
          </div>
          <div className="selected-price">
            <strong>{priceText(shownPrice(stock))}</strong>
            <Change value={shownChange(stock)} />
          </div>
          <Suspense
            fallback={
              <div className="price-chart chart-loading">
                Đang tải biểu đồ...
              </div>
            }
          >
            <PriceChart history={history} error={error} symbol={stock.symbol} />
          </Suspense>
        </div>
        <div className="analysis-side">
          <div className="ml-summary">
            <div>
              <span>ĐIỂM ML HIỆN TẠI</span>
              <strong>
                {stock.ml_score == null ? "—" : format.format(stock.ml_score)}
                <small> / 100</small>
              </strong>
            </div>
            <div>
              <span>THỨ HẠNG TRONG VN30</span>
              <strong>
                #{comparison?.rank ?? "—"}
                <small> / {comparison?.ranked_count ?? 30}</small>
              </strong>
            </div>
          </div>
          {stock.explanation && (
            <section
              className="score-explanation"
              aria-label="Cách tính điểm ML"
            >
              <h3>
                Vì sao {stock.symbol} được{" "}
                {format.format(stock.explanation.score)} điểm?
              </h3>
              <p className="score-intro">
                Điểm 50 là trung tính.{" "}
                {signalGap === 0
                  ? "Tín hiệu vượt và kém VNINDEX đang cân bằng."
                  : `Tín hiệu ${signalGap < 0 ? "kém" : "vượt"} VNINDEX nhiều hơn tín hiệu ${signalGap < 0 ? "vượt" : "kém"} ${format.format(Math.abs(signalGap))} điểm phần trăm, nên điểm ${signalGap < 0 ? "dưới" : "trên"} 50.`}
              </p>
              <p className="score-context">
                Mô hình đối chiếu lợi nhuận của {stock.symbol} với VNINDEX sau
                20 phiên giao dịch:
              </p>
              <div className="signal-bar" aria-hidden="true">
                <span
                  className="below"
                  style={{ width: `${stock.explanation.below}%` }}
                />
                <span
                  className="neutral"
                  style={{ width: `${stock.explanation.neutral}%` }}
                />
                <span
                  className="above"
                  style={{ width: `${stock.explanation.above}%` }}
                />
              </div>
              <div className="signal-legend">
                <div>
                  <i className="below" />
                  <span>
                    Kém VNINDEX từ {format.format(threshold)} điểm phần trăm
                  </span>
                  <b>{format.format(stock.explanation.below)}%</b>
                </div>
                <div>
                  <i className="neutral" />
                  <span>
                    Gần VNINDEX (trong ±{format.format(threshold)} điểm phần
                    trăm)
                  </span>
                  <b>{format.format(stock.explanation.neutral)}%</b>
                </div>
                <div>
                  <i className="above" />
                  <span>
                    Vượt VNINDEX từ {format.format(threshold)} điểm phần trăm
                  </span>
                  <b>{format.format(stock.explanation.above)}%</b>
                </div>
              </div>
              <div className="score-formula">
                <span>
                  50 + ({format.format(stock.explanation.above)} −{" "}
                  {format.format(stock.explanation.below)}) ÷ 2
                </span>
                <strong>= {format.format(stock.explanation.score)} điểm</strong>
              </div>
              <p className="score-note">
                Nhóm trong ngưỡng giữ điểm gần 50; chỉ phần chênh giữa hai nhóm
                còn lại làm điểm thay đổi.
              </p>
              <details className="score-method">
                <summary>Mô hình tạo các tỷ lệ này ra sao?</summary>
                <p>
                  Ví dụ: mã tăng 5%, VNINDEX tăng 1% thì mã vượt chỉ số 4 điểm
                  phần trăm. Ngưỡng {format.format(threshold)} điểm phần trăm
                  gồm 3 điểm phần trăm mục tiêu chênh lệch và 0,6 điểm phần trăm
                  chi phí giao dịch giả định. Random Forest và XGBoost dùng 19
                  chỉ báo của phiên đã chốt; tín hiệu hai mô hình được lấy trung
                  bình rồi điều chỉnh theo tỷ lệ ba nhóm trong dữ liệu huấn
                  luyện. Các tỷ lệ cộng lại thành 100%, nhưng không phải xác
                  suất sinh lời hay mức đóng góp của từng chỉ báo.
                </p>
              </details>
            </section>
          )}
          <div className="alternatives">
            <div className="alternatives-heading">
              <span>
                {comparison?.better_available
                  ? "MÃ CÓ ĐIỂM CAO HƠN"
                  : "MÃ GẦN NHẤT ĐỂ ĐỐI CHIẾU"}
              </span>
              <Sparkles size={15} />
            </div>
            {comparison?.alternatives?.map((item) => (
              <button
                type="button"
                key={item.symbol}
                onClick={() => onSelect(item.symbol)}
              >
                <span className="alternative-name">{item.symbol}</span>
                <span>ML {format.format(item.ml_score)}</span>
                <b className={tone(item.score_gap)}>
                  {signedText(item.score_gap, "đ")}
                </b>
                <ChevronRight size={15} />
              </button>
            ))}
            {!comparison && (
              <p className="subtle">
                {error || "Đang so sánh với các mã khác..."}
              </p>
            )}
          </div>
          <p className="analysis-disclaimer">
            <CircleHelp size={15} /> Điểm ML phản ánh tín hiệu vượt/kém VNINDEX,
            không phải xác suất sinh lời hay lệnh mua.
          </p>
        </div>
      </div>
      <ProfitLossEstimator
        key={stock.symbol}
        symbol={stock.symbol}
        history={history}
        error={error}
      />
    </section>
  );
}
