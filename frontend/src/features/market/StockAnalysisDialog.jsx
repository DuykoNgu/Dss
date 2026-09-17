import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import StockAnalysis from "./StockAnalysis";

export default function StockAnalysisDialog({
  stock,
  onClose,
  onSelect,
  onAsk,
}) {
  const dialogRef = useRef(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialog.showModal();
    return () => {
      dialog.close();
      document.body.style.overflow = previousOverflow;
    };
  }, []);

  function closeOnBackdrop(event) {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom
    ) {
      onClose();
    }
  }

  return (
    <dialog
      ref={dialogRef}
      className="stock-dialog"
      aria-label={`Phân tích mã ${stock.symbol}`}
      onClick={closeOnBackdrop}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      <button
        className="stock-dialog-close"
        type="button"
        aria-label="Đóng phân tích"
        autoFocus
        onClick={onClose}
      >
        <X size={21} />
      </button>
      <div className="stock-dialog-scroll">
        <StockAnalysis stock={stock} onSelect={onSelect} onAsk={onAsk} />
      </div>
    </dialog>
  );
}
