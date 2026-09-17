import { useEffect, useRef, useState } from "react";
import { MessageCircle, Send, Sparkles, X } from "lucide-react";
import { askAssistant } from "../../common/api";
import ChatMessage from "./ChatMessage";

const initialMessage = {
  role: "assistant",
  text: "Xin chào! Tôi chỉ trả lời về xếp hạng ML của VN30, một mã cụ thể, so sánh hai mã và cách mô hình hoạt động.",
};
const suggestions = [
  "Mã nào được ML xếp cao nhất?",
  "So sánh FPT và HPG",
  "Điểm ML hoạt động thế nào?",
];

export default function ChatWidget({
  open,
  setOpen,
  draft,
  setDraft,
  onSelect,
}) {
  const [messages, setMessages] = useState([initialMessage]);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);
  const bottomRef = useRef(null);
  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);
  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [open, messages]);
  useEffect(() => {
    if (!open) return;
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [open, setOpen]);
  async function submit(question) {
    const text = question.trim();
    if (!text || busy) return;
    setDraft("");
    setMessages((current) => [...current, { role: "user", text }]);
    setBusy(true);
    try {
      const result = await askAssistant(text);
      setMessages((current) => [...current, { role: "assistant", result }]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", text: error.message },
      ]);
    } finally {
      setBusy(false);
    }
  }
  function choose(symbol) {
    onSelect(symbol);
    setOpen(false);
  }
  return (
    <>
      {open && <div className="chat-backdrop" onClick={() => setOpen(false)} />}
      {open && (
        <section
          className="chat-panel"
          role="dialog"
          aria-label="Trợ lý ML VN30"
        >
          <div className="chat-header">
            <span className="chat-avatar">
              <Sparkles size={18} />
            </span>
            <div>
              <strong>Trợ lý ML</strong>
              <small>
                <span /> Chỉ trong phạm vi VN30
              </small>
            </div>
            <button
              type="button"
              className="chat-close"
              aria-label="Đóng trợ lý"
              onClick={() => setOpen(false)}
            >
              <X size={19} />
            </button>
          </div>
          <div className="chat-messages">
            <div className="chat-date">PHIÊN TRAO ĐỔI NÀY</div>
            {messages.map((message, index) => (
              <ChatMessage key={index} message={message} onSelect={choose} />
            ))}
            {busy && (
              <div className="chat-message assistant">
                <div className="chat-bubble typing">
                  <i />
                  <i />
                  <i />
                </div>
              </div>
            )}
            {messages.length === 1 && (
              <div className="chat-suggestions">
                {suggestions.map((question) => (
                  <button
                    type="button"
                    key={question}
                    onClick={() => submit(question)}
                  >
                    {question}
                  </button>
                ))}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
          <form
            className="chat-input"
            onSubmit={(event) => {
              event.preventDefault();
              submit(draft);
            }}
          >
            <input
              ref={inputRef}
              aria-label="Hỏi trợ lý ML"
              placeholder="Hỏi về một mã VN30..."
              maxLength="240"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
            />
            <button
              type="submit"
              disabled={!draft.trim() || busy}
              aria-label="Gửi câu hỏi"
            >
              <Send size={17} />
            </button>
          </form>
          <div className="chat-foot">
            Trả lời dựa trên dữ liệu & quy tắc ML của dự án.
          </div>
        </section>
      )}
      <button
        className={`chat-launcher ${open ? "is-open" : ""}`}
        type="button"
        onClick={() => setOpen(!open)}
        aria-label={open ? "Đóng trợ lý ML" : "Mở trợ lý ML"}
        aria-expanded={open}
      >
        <span className="launcher-icon">
          {open ? <X size={23} /> : <MessageCircle size={23} />}
        </span>
        <span className="launcher-label">
          {open ? "Đóng chat" : "Hỏi trợ lý ML"}
        </span>
        {!open && <span className="launcher-pulse" />}
      </button>
    </>
  );
}
