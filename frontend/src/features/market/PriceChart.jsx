import { useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { dateText, priceText, signedText, tone } from "../../common/format";

export default function PriceChart({ history, error, symbol }) {
  const [period, setPeriod] = useState(60);
  const chart = history.slice(-period);
  const periodReturn =
    chart.length > 1 ? (chart.at(-1).close / chart[0].close - 1) * 100 : null;
  return (
    <>
      <div className="chart-header">
        <div>
          <span>DIỄN BIẾN GIÁ ĐÓNG CỬA</span>
          <b className={tone(periodReturn)}>
            {periodReturn == null ? "—" : signedText(periodReturn)}
          </b>
        </div>
        <div className="periods" aria-label="Khoảng thời gian biểu đồ">
          <button
            type="button"
            className={period === 20 ? "active" : ""}
            onClick={() => setPeriod(20)}
          >
            1T
          </button>
          <button
            type="button"
            className={period === 60 ? "active" : ""}
            onClick={() => setPeriod(60)}
          >
            3T
          </button>
          <button
            type="button"
            className={period === 120 ? "active" : ""}
            onClick={() => setPeriod(120)}
          >
            6T
          </button>
        </div>
      </div>
      <div
        className="price-chart"
        role="img"
        aria-label={`Đồ thị giá đóng cửa ${symbol} theo thời gian`}
      >
        {history.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={chart}
              margin={{ top: 10, right: 2, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#168a78" stopOpacity={0.23} />
                  <stop offset="100%" stopColor="#168a78" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                vertical={false}
                stroke="#e8eeea"
                strokeDasharray="3 5"
              />
              <XAxis
                dataKey="date"
                tickFormatter={(value) => value.slice(5)}
                tick={{ fill: "#8a9a94", fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                minTickGap={24}
              />
              <YAxis
                domain={["auto", "auto"]}
                tick={{ fill: "#8a9a94", fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                width={42}
              />
              <Tooltip
                labelFormatter={dateText}
                formatter={(value) => [priceText(value), "Đóng cửa"]}
                contentStyle={{
                  borderRadius: 10,
                  border: "1px solid #dce7df",
                  fontSize: 12,
                }}
              />
              <Area
                dataKey="close"
                type="monotone"
                stroke="#137d6f"
                strokeWidth={2.4}
                fill="url(#priceFill)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="chart-loading">
            {error || "Đang tải lịch sử giá..."}
          </div>
        )}
      </div>
    </>
  );
}
