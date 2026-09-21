export const EXPERIENCE = {
  new: {
    title: "Tôi mới bắt đầu",
    description: "Cần hướng dẫn rõ ràng và giải thích thuật ngữ.",
  },
  experienced: {
    title: "Tôi đã có kinh nghiệm",
    description: "Đã từng phân tích hoặc giao dịch cổ phiếu.",
  },
};

export const KNOWLEDGE = {
  starter: {
    title: "Đang làm quen",
    description: "Chưa tự tin đọc chỉ số và tín hiệu thị trường.",
  },
  practical: {
    title: "Nắm kiến thức cơ bản",
    description: "Hiểu giá, xu hướng và một số chỉ báo phổ biến.",
  },
  advanced: {
    title: "Có thể tự phân tích",
    description: "Đã có phương pháp và muốn thêm dữ liệu kiểm chứng.",
  },
};

export const GOALS = {
  learn: {
    title: "Học cách đọc tín hiệu",
    description: "Hiểu dữ liệu và lý do phía sau mỗi nhận định.",
    result:
      "giải thích tín hiệu bằng ngôn ngữ dễ hiểu trước khi đi sâu vào số liệu",
  },
  discover: {
    title: "Tìm mã đáng nghiên cứu",
    description: "Sàng lọc nhanh các cơ hội trong rổ VN30.",
    result: "ưu tiên bảng xếp hạng và các mã đáng nghiên cứu trong VN30",
  },
  verify: {
    title: "Kiểm chứng quyết định",
    description: "Đối chiếu nhận định cá nhân với Rule Score và ML Score.",
    result:
      "ưu tiên bằng chứng, rủi ro và sự đồng thuận giữa Rule Score với ML Score",
  },
  market: {
    title: "Theo dõi thị trường",
    description: "Nắm nhanh trạng thái và diễn biến của nhóm VN30.",
    result: "ưu tiên bức tranh thị trường và các thay đổi đáng chú ý của VN30",
  },
};

export function profileSummary({ experience, knowledge, goal }) {
  if (!EXPERIENCE[experience] || !KNOWLEDGE[knowledge] || !GOALS[goal]) {
    return "Hãy hoàn tất ba lựa chọn để DSS xác định mục tiêu phù hợp.";
  }
  const tone =
    experience === "new" || knowledge === "starter"
      ? "giải thích từ nền tảng và "
      : "đi thẳng vào dữ liệu và ";
  return `DSS sẽ ${tone}${GOALS[goal].result}.`;
}
