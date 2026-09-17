import { ArrowRight, ChevronRight } from "lucide-react";
import { format } from "../../common/format";

export default function ChatMessage({ message, onSelect }) {
  const result = message.result;
  const items =
    result?.kind === "top"
      ? result.items
      : result?.kind === "comparison"
        ? result.items.map((entry) => entry.selected)
        : [];
  const selected = result?.recommendation?.selected;
  const answer =
    result?.message ||
    (selected &&
      `${selected.symbol}: ML ${format.format(selected.ml_score)}/100, hạng ${result.recommendation.rank}/${result.recommendation.ranked_count}.`);

  return (
    <div className={`chat-message ${message.role}`}>
      <div className="chat-bubble">
        {message.text && <p>{message.text}</p>}
        {answer && <p>{answer}</p>}
        {result?.kind === "blocked" && (
          <small>Thử một câu hỏi mẫu bên dưới.</small>
        )}
        {items.length > 0 && (
          <div className="chat-results">
            {items.map((item) => (
              <button
                key={item.symbol}
                type="button"
                onClick={() => onSelect(item.symbol)}
              >
                <span>{item.symbol}</span>
                <b>ML {format.format(item.ml_score)}</b>
                <ChevronRight size={14} />
              </button>
            ))}
          </div>
        )}
        {selected && (
          <button
            className="chat-action"
            type="button"
            onClick={() => onSelect(selected.symbol)}
          >
            Xem phân tích {selected.symbol} <ArrowRight size={14} />
          </button>
        )}
      </div>
    </div>
  );
}
