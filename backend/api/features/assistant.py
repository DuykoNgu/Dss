"""VN30 assistant rules and API route."""

from __future__ import annotations

import re
import unicodedata

from api.features.market import MarketState, recommendation

QUESTION_LIMIT = 240


def normalized(text: str) -> str:
    plain = unicodedata.normalize("NFD", text.lower())
    return "".join(character for character in plain if unicodedata.category(character) != "Mn").replace("đ", "d")


def answer_question(question: str, rows: list[dict]) -> dict:
    question = question.strip()
    if not question or len(question) > QUESTION_LIMIT:
        raise ValueError(f"Câu hỏi phải có từ 1 đến {QUESTION_LIMIT} ký tự.")
    members = {row["symbol"] for row in rows}
    explicit_codes = re.findall(r"(?<![A-Za-z0-9])[A-Z]{2,5}(?![A-Za-z0-9])", question)
    unknown_codes = [code for code in explicit_codes if code not in members and code not in {"VN30", "ML", "DSS", "RF"}]
    if unknown_codes:
        return {"kind": "blocked", "message": f"{unknown_codes[0]} không thuộc 30 mã VN30 đang theo dõi. Tôi chỉ trả lời trong phạm vi này."}

    text = " ".join(re.findall(r"[a-z0-9]+", normalized(question)))
    comparison = re.fullmatch(r"so sanh (?:ma )?([a-z]{2,5}) (?:va|voi) (?:ma )?([a-z]{2,5})(?: theo ml)?", text)
    if (comparison and comparison.group(1) != comparison.group(2)
            and all(code.upper() in members for code in comparison.groups())):
        pair = [recommendation(rows, code.upper()) for code in comparison.groups()]
        pair.sort(key=lambda item: item["selected"]["ml_score"], reverse=True)
        message = (f"Trong hai mã, ML xếp {pair[0]['selected']['symbol']} cao hơn theo điểm hiện tại."
                   if pair[0]["selected"]["ml_score"] > pair[1]["selected"]["ml_score"]
                   else "Hai mã đang đồng điểm ML ở phiên dữ liệu này.")
        return {"kind": "comparison", "items": pair, "message": message}
    symbol_patterns = (
        r"([a-z]{2,5}) the nao(?: theo ml)?",
        r"(?:diem ml cua|xep hang|so sanh) (?:ma )?([a-z]{2,5})",
        r"(?:toi )?(?:muon dau tu|nen dau tu|chon) (?:vao )?(?:ma )?([a-z]{2,5})",
    )
    for pattern in symbol_patterns:
        match = re.fullmatch(pattern, text)
        if match and match.group(1).upper() in members:
            return {"kind": "symbol", "recommendation": recommendation(rows, match.group(1).upper())}
    top_patterns = (
        r"(?:toi )?(?:khong|chua) biet (?:chon|dau tu) ma nao",
        r"(?:toi )?(?:khong|chua) biet dau tu gi",
        r"ma nao (?:duoc )?ml xep (?:cao nhat|hang dau)",
        r"(?:goi y|top|xep hang) (?:ma )?vn30",
        r"nen chon ma nao(?: trong vn30)?",
    )
    if any(re.fullmatch(pattern, text) for pattern in top_patterns):
        top = sorted((row for row in rows if row["ml_score"] is not None),
                     key=lambda row: (-row["ml_score"], row["symbol"]))[:5]
        return {"kind": "top", "items": top, "message": "5 mã VN30 được mô hình xếp cao nhất ở phiên dữ liệu hiện tại."}
    if text in {"ml la gi", "diem ml la gi", "diem ml hoat dong the nao", "mo hinh ml hoat dong the nao", "do tin cay cua mo hinh ml"}:
        return {"kind": "method", "message": "ML dùng Random Forest + XGBoost và 19 đặc trưng kỹ thuật để xếp hạng tương đối 30 mã VN30 theo nhãn lợi nhuận vượt VNINDEX sau 20 phiên. Điểm 0–100 chỉ là điểm xếp hạng, không phải xác suất sinh lời hay lệnh mua. Backtest quá khứ không bảo đảm kết quả tương lai."}
    return {"kind": "blocked", "message": "Tôi chỉ hỗ trợ hỏi về xếp hạng ML của VN30, một mã VN30, so sánh mã và cách mô hình hoạt động. Hãy thử: ‘Mã nào được ML xếp cao nhất?’"}


def ask(state: MarketState, body: dict) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("question"), str):
        raise ValueError("Câu hỏi không hợp lệ.")
    return answer_question(body["question"], state.market()["stocks"])


POST_ROUTES = {"/api/ask": ask}
