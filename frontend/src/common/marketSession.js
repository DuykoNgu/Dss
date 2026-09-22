import dayjs from "dayjs";
import timezone from "dayjs/plugin/timezone.js";
import utc from "dayjs/plugin/utc.js";

dayjs.extend(utc);
dayjs.extend(timezone);

const MARKET_TIME_ZONE = "Asia/Ho_Chi_Minh";
const DAILY_CANDLE_READY_HOUR = 15;

export function latestClosedSessionDate(now = dayjs()) {
  let session = dayjs(now).tz(MARKET_TIME_ZONE);
  if (session.hour() < DAILY_CANDLE_READY_HOUR) {
    session = session.subtract(1, "day");
  }
  while (session.day() === 0 || session.day() === 6) {
    session = session.subtract(1, "day");
  }
  return session.format("YYYY-MM-DD");
}
