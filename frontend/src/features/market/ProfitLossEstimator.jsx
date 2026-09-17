import { useEffect, useState } from "react";
import { getProfitLoss } from "../../common/api";
import { dateText, format, signedText, tone, whole } from "../../common/format";
import "./profit-loss.css";

export default function ProfitLossEstimator({
  symbol,
  history,
  error: historyError,
}) {
  const [entryDate, setEntryDate] = useState("");
  const [buyPrice, setBuyPrice] = useState("");
  const [quantity, setQuantity] = useState("100");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!history.length) return;
    const initial = history.at(-20) ?? history[0];
    setEntryDate(initial.date);
    setBuyPrice(String(initial.close));
    setResult(null);
  }, [history]);

  async function calculate(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(
        await getProfitLoss({
          symbol,
          entry_date: entryDate,
          buy_price: Number(buyPrice),
          quantity: Number(quantity),
        }),
      );
    } catch (reason) {
      setError(reason.message);
    } finally {
      setLoading(false);
    }
  }

  function chooseDate(event) {
    const date = event.target.value;
    setEntryDate(date);
    setBuyPrice(String(history.find((candle) => candle.date === date).close));
    setResult(null);
  }

  const sessions = result?.sessions ?? [];
  const latest = sessions.at(-1);
  const bestProfit = Math.max(0, ...sessions.map((row) => row.pnl_vnd));
  const worstLoss = Math.min(0, ...sessions.map((row) => row.pnl_vnd));

  return (
    <section
      id="profit-loss"
      className="profit-loss"
      aria-label={`Ước tính lời lỗ ${symbol}`}
    >
      <div className="profit-loss-heading">
        <div>
          <span className="section-kicker">MÔ PHỎNG VỊ THẾ</span>
          <h3>Ước tính lời/lỗ sau mỗi phiên</h3>
        </div>
        <p>Dựa trên giá đóng cửa đã có · Không dự báo giá tương lai</p>
      </div>
      {history.length ? (
        <>
          <form className="profit-loss-form" onSubmit={calculate}>
            <label>
              Ngày mua giả định
              <select value={entryDate} onChange={chooseDate} required>
                {[...history].reverse().map((candle) => (
                  <option key={candle.date} value={candle.date}>
                    {dateText(candle.date)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Giá mua (nghìn đồng/CP)
              <input
                type="number"
                min="0.01"
                max="1000"
                step="0.01"
                value={buyPrice}
                onChange={(event) => {
                  setBuyPrice(event.target.value);
                  setResult(null);
                }}
                required
              />
            </label>
            <label>
              Số cổ phiếu
              <input
                type="number"
                min="1"
                max="10000000"
                step="1"
                value={quantity}
                onChange={(event) => {
                  setQuantity(event.target.value);
                  setResult(null);
                }}
                required
              />
            </label>
            <button type="submit" disabled={loading}>
              {loading ? "Đang tính..." : "Tính lời/lỗ"}
            </button>
          </form>
          {error && (
            <p className="profit-loss-error" role="alert">
              {error}
            </p>
          )}
          {latest && (
            <div className="profit-loss-results" aria-live="polite">
              <div className="profit-loss-summary">
                <div>
                  <span>PHIÊN {dateText(latest.date)} · TẠM TÍNH</span>
                  <strong className={tone(latest.pnl_vnd)}>
                    {signedText(latest.pnl_vnd, " đ")}
                  </strong>
                  <small className={tone(latest.pnl_pct)}>
                    {signedText(latest.pnl_pct)} trên tổng tiền mua
                  </small>
                </div>
                <div>
                  <span>MỨC LỜI CAO NHẤT</span>
                  <strong className={bestProfit ? "up" : "flat"}>
                    {bestProfit
                      ? signedText(bestProfit, " đ")
                      : "Chưa có phiên lời"}
                  </strong>
                </div>
                <div>
                  <span>MỨC LỖ LỚN NHẤT</span>
                  <strong className={worstLoss ? "down" : "flat"}>
                    {worstLoss
                      ? signedText(worstLoss, " đ")
                      : "Chưa có phiên lỗ"}
                  </strong>
                </div>
              </div>
              <p className="profit-loss-costs">
                Tiền mua {whole.format(result.purchase_value_vnd)} đ + chi phí
                mua {whole.format(result.buy_costs_vnd)} đ ={" "}
                {whole.format(result.buy_total_vnd)} đ. Chi phí bán giả định
                phiên gần nhất: {whole.format(latest.sell_costs_vnd)} đ.
              </p>
              <div className="profit-loss-table-wrap">
                <table className="profit-loss-table">
                  <thead>
                    <tr>
                      <th>Phiên</th>
                      <th>Đóng cửa</th>
                      <th>So với phiên trước</th>
                      <th>Lời/lỗ tạm tính</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...sessions].reverse().map((row) => (
                      <tr key={row.date}>
                        <td>{dateText(row.date)}</td>
                        <td>{format.format(row.close)} nghìn đ</td>
                        <td className={tone(row.daily_change_vnd)}>
                          {row.daily_change_vnd == null
                            ? "—"
                            : signedText(row.daily_change_vnd, " đ")}
                        </td>
                        <td className={tone(row.pnl_vnd)}>
                          <b>{signedText(row.pnl_vnd, " đ")}</b>
                          <small>{signedText(row.pnl_pct)}</small>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <details className="profit-loss-method">
                <summary>Cách tính và giả định chi phí</summary>
                <p>
                  Lời/lỗ tạm tính = giá đóng cửa × số cổ phiếu × 1.000 − chi phí
                  bán giả định − tổng tiền mua. Tổng tiền mua đã gồm giá mua ×
                  số cổ phiếu × 1.000 và chi phí mua. Chi phí mua{" "}
                  {format.format(result.buy_cost_rate_pct)}%; chi phí bán{" "}
                  {format.format(result.sell_cost_rate_pct)}% gồm phí, thuế và
                  trượt giá giả định. Giá đóng cửa chỉ để định giá; đây không
                  phải giao dịch đã thực hiện. Chưa tính cổ tức và quyền cổ
                  phiếu.
                </p>
              </details>
            </div>
          )}
        </>
      ) : (
        <p className="profit-loss-empty">
          {historyError || "Đang tải các phiên đóng cửa để tính lời/lỗ..."}
        </p>
      )}
    </section>
  );
}
