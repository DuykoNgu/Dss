export const format = new Intl.NumberFormat("vi-VN", {
  maximumFractionDigits: 2,
});
export const icFormat = new Intl.NumberFormat("vi-VN", {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3,
});
export const whole = new Intl.NumberFormat("vi-VN", {
  maximumFractionDigits: 0,
});

export const dateText = (date) =>
  date ? date.split("-").reverse().join("/") : "—";

export const priceText = (value) =>
  value == null ? "—" : format.format(value);

export const signedText = (value, suffix = "%") =>
  value == null
    ? "—"
    : `${value > 0 ? "+" : ""}${format.format(value)}${suffix}`;

export const tone = (value) => (value > 0 ? "up" : value < 0 ? "down" : "flat");

export const shownPrice = (stock) => stock.quote?.price ?? stock.close;
export const shownChange = (stock) =>
  stock.quote?.change_pct ?? stock.change_pct;
export const shownVolume = (stock) => stock.quote?.volume ?? stock.volume;
export const shownOpen = (stock) => stock.quote?.open ?? stock.open;
export const shownHigh = (stock) => stock.quote?.high ?? stock.high;
export const shownLow = (stock) => stock.quote?.low ?? stock.low;
export const shownReference = (stock) =>
  stock.quote?.reference ?? stock.previous_close;
