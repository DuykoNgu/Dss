import { useMemo, useState } from "react";
import { ChevronRight, Filter, Search, ShieldCheck } from "lucide-react";
import Change from "../../common/Change";
import {
  format,
  priceText,
  shownChange,
  shownHigh,
  shownLow,
  shownOpen,
  shownPrice,
  shownReference,
  shownVolume,
  tone,
  whole,
} from "../../common/format";

function compareStocks(a, b, order) {
  switch (order) {
    case "score":
      return (b.ml_score ?? -Infinity) - (a.ml_score ?? -Infinity);
    case "change":
      return (shownChange(b) ?? -Infinity) - (shownChange(a) ?? -Infinity);
    case "volume":
      return shownVolume(b) - shownVolume(a);
    default:
      return a.symbol.localeCompare(b.symbol);
  }
}

export default function MarketBoard({
  stocks,
  selectedSymbol,
  onSelect,
  model,
  modelVersion,
}) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [sort, setSort] = useState("symbol");
  const visible = useMemo(() => {
    const query = search.trim().toUpperCase();
    return stocks
      .filter(
        (stock) =>
          stock.symbol.includes(query) &&
          (filter === "all" || tone(shownChange(stock)) === filter),
      )
      .sort((a, b) => compareStocks(a, b, sort));
  }, [stocks, search, filter, sort]);
  return (
    <section
      className="market-card"
      id="thi-truong"
      aria-labelledby="board-heading"
    >
      <div className="card-heading">
        <div>
          <span className="section-kicker">
            <span className="kicker-square" /> BẢNG GIÁ / VN30
          </span>
          <h2 id="board-heading">Bảng giá VN30</h2>
          <p>Chọn một mã để mở phân tích, đồ thị và so sánh điểm ML.</p>
        </div>
        <span className="model-chip">
          <span /> ML · {model?.label_strategy?.toUpperCase() || "—"} T+
          {model?.horizon ?? "—"}
          {modelVersion && (
            <small title={`Model ${modelVersion}`}> · {modelVersion}</small>
          )}
        </span>
      </div>
      <div className="board-controls">
        <div className="board-search">
          <Search size={17} />
          <input
            aria-label="Tìm mã VN30"
            placeholder="Tìm mã cổ phiếu..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
        <div className="filter-group" aria-label="Lọc biến động">
          <Filter size={15} />
          <button
            type="button"
            className={filter === "all" ? "chosen" : ""}
            onClick={() => setFilter("all")}
          >
            Tất cả
          </button>
          <button
            type="button"
            className={filter === "up" ? "chosen" : ""}
            onClick={() => setFilter("up")}
          >
            Tăng
          </button>
          <button
            type="button"
            className={filter === "down" ? "chosen" : ""}
            onClick={() => setFilter("down")}
          >
            Giảm
          </button>
        </div>
        <label className="sort-control">
          Sắp xếp{" "}
          <select
            value={sort}
            onChange={(event) => setSort(event.target.value)}
          >
            <option value="symbol">Mã A–Z</option>
            <option value="score">Điểm ML cao nhất</option>
            <option value="change">Biến động mạnh nhất</option>
            <option value="volume">Khối lượng lớn nhất</option>
          </select>
        </label>
      </div>
      <div
        className="table-scroll"
        role="region"
        aria-label="Bảng giá VN30, cuộn ngang để xem thêm cột"
        tabIndex="0"
      >
        <table>
          <thead>
            <tr>
              <th>MÃ</th>
              <th>ĐÓNG CỬA TRƯỚC</th>
              <th>MỞ CỬA</th>
              <th>CAO NHẤT</th>
              <th>THẤP NHẤT</th>
              <th>GIÁ / ĐÓNG CỬA</th>
              <th>THAY ĐỔI</th>
              <th>KHỐI LƯỢNG</th>
              <th>ĐIỂM ML</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {visible.map((stock) => (
              <tr
                key={stock.symbol}
                className={
                  selectedSymbol === stock.symbol ? "selected-row" : ""
                }
              >
                <td className="symbol-cell">
                  <button
                    type="button"
                    onClick={() => onSelect(stock.symbol)}
                    aria-label={`Phân tích mã ${stock.symbol}`}
                  >
                    <span className="symbol-avatar">
                      {stock.symbol.slice(0, 1)}
                    </span>
                    <span>{stock.symbol}</span>
                  </button>
                </td>
                <td>{priceText(shownReference(stock))}</td>
                <td>{priceText(shownOpen(stock))}</td>
                <td>{priceText(shownHigh(stock))}</td>
                <td>{priceText(shownLow(stock))}</td>
                <td className={`close-price ${tone(shownChange(stock))}`}>
                  {priceText(shownPrice(stock))}
                </td>
                <td className={`change-cell ${tone(shownChange(stock))}`}>
                  <Change value={shownChange(stock)} compact />
                </td>
                <td>{whole.format(shownVolume(stock))}</td>
                <td>
                  <span
                    className={`score-pill ${stock.ml_score == null ? "unavailable" : ""}`}
                    style={{
                      "--score": `${Math.max(0, Math.min(stock.ml_score ?? 0, 100))}%`,
                    }}
                  >
                    {stock.ml_score == null
                      ? "—"
                      : format.format(stock.ml_score)}
                  </span>
                </td>
                <td>
                  <button
                    className="row-action"
                    type="button"
                    disabled={stock.ml_score == null}
                    onClick={() => onSelect(stock.symbol)}
                    aria-label={`Xem phân tích ${stock.symbol}`}
                  >
                    <ChevronRight size={16} />
                  </button>
                </td>
              </tr>
            ))}
            {!visible.length && (
              <tr>
                <td className="no-results" colSpan="10">
                  Không tìm thấy mã VN30 phù hợp.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="board-bottom">
        <span>
          Hiển thị <strong>{visible.length}</strong> / {stocks.length} mã VN30
        </span>
        <span>
          <ShieldCheck size={14} /> Giá nến cuối ngày · Biến động so với đóng
          cửa trước
        </span>
      </div>
    </section>
  );
}
