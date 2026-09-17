import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { signedText, tone } from "./format";

export default function Change({ value, compact = false }) {
  const Icon = value < 0 ? ArrowDownRight : ArrowUpRight;
  return (
    <span className={`change ${tone(value)}`}>
      {value !== 0 && value != null && (
        <Icon size={compact ? 13 : 16} strokeWidth={2.2} />
      )}
      {signedText(value)}
    </span>
  );
}
