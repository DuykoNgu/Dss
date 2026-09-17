import { ShieldCheck } from "lucide-react";
import { icFormat, signedText } from "../../common/format";

export default function Research({ research }) {
  return (
    <section className="research" id="kiem-chung">
      <div className="research-copy">
        <span className="section-kicker">
          <span className="kicker-square" /> KIỂM CHỨNG / WALK-FORWARD
        </span>
        <h2>
          Đặt mô hình trong
          <br />
          <em>đúng bối cảnh.</em>
        </h2>
        <p>
          Điểm hiện tại được tạo bởi mô hình ML chung với nhãn lợi nhuận vượt
          VNINDEX sau 20 phiên. Các con số dưới đây là thử nghiệm quá khứ, không
          phải lời hứa về lợi nhuận tương lai.
        </p>
        <div className="research-note">
          <ShieldCheck size={18} />
          <span>
            Thành phần VN30 được xét theo từng kỳ trong backtest; dữ liệu giá
            trên trang là cuối phiên, không trực tiếp.
          </span>
        </div>
      </div>
      <div className="research-numbers">
        {research ? (
          <>
            <div>
              <span>RANK IC TRUNG BÌNH</span>
              <strong>
                {research.rank_ic > 0 ? "+" : ""}
                {icFormat.format(research.rank_ic)}
              </strong>
              <small>Dương trong {research.positive_years} năm được đo</small>
            </div>
            <div>
              <span>DANH MỤC ML · TOÀN KỲ</span>
              <strong>{signedText(research.portfolio_return)}</strong>
              <small>
                Nắm đều rổ: {signedText(research.equal_weight_return)}
              </small>
            </div>
            <div>
              <span>SỤT GIẢM TỐI ĐA</span>
              <strong className="risk-number">
                {signedText(research.max_drawdown)}
              </strong>
              <small>Rủi ro đã ghi nhận trong backtest</small>
            </div>
          </>
        ) : (
          <p>Chưa có báo cáo kiểm chứng trên máy chủ này.</p>
        )}
      </div>
    </section>
  );
}
