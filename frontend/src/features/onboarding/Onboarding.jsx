import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BarChart3,
  BookOpen,
  Check,
  Compass,
  GraduationCap,
  LineChart,
  SearchCheck,
  Sparkles,
} from "lucide-react";
import { EXPERIENCE, GOALS, KNOWLEDGE, profileSummary } from "./profile";

const steps = ["experience", "knowledge", "goal", "summary"];
const icons = {
  new: GraduationCap,
  experienced: LineChart,
  starter: BookOpen,
  practical: BarChart3,
  advanced: SearchCheck,
  learn: BookOpen,
  discover: Compass,
  verify: SearchCheck,
  market: BarChart3,
};

function Choice({ id, item, selected, onSelect }) {
  const Icon = icons[id];
  return (
    <button
      type="button"
      className={`onboarding-choice${selected ? " selected" : ""}`}
      onClick={() => onSelect(id)}
      aria-pressed={selected}
    >
      <span className="choice-icon">
        <Icon size={22} />
      </span>
      <span className="choice-copy">
        <strong>{item.title}</strong>
        <small>{item.description}</small>
      </span>
      <span className="choice-check" aria-hidden="true">
        <Check size={15} />
      </span>
    </button>
  );
}

export default function Onboarding({ onComplete, onSkip }) {
  const shell = useRef(null);
  const [step, setStep] = useState(0);
  const [profile, setProfile] = useState({
    experience: "",
    knowledge: "",
    goal: "",
  });
  const key = steps[step];
  const options =
    key === "experience" ? EXPERIENCE : key === "knowledge" ? KNOWLEDGE : GOALS;
  const prompts = {
    experience: {
      title: "Bạn đang ở đâu trên hành trình đầu tư?",
      body: "Chọn mô tả gần với bạn nhất để DSS điều chỉnh cách trình bày.",
    },
    knowledge: {
      title: "Bạn hiểu thị trường đến đâu?",
      body:
        profile.experience === "new"
          ? "Không cần biết thuật ngữ chuyên môn. Hãy chọn mức bạn thấy thoải mái."
          : "Chọn mức phản ánh cách bạn đang đọc và đánh giá một cổ phiếu.",
    },
    goal: {
      title: "Hôm nay bạn muốn giải quyết việc gì?",
      body: "DSS sẽ ưu tiên nội dung phù hợp với mục đích chính này.",
    },
  };
  const canContinue = key === "summary" || Boolean(profile[key]);

  useEffect(() => {
    shell.current?.scrollTo({ top: 0 });
  }, [step]);

  function choose(value) {
    setProfile((current) => ({ ...current, [key]: value }));
  }

  return (
    <div
      ref={shell}
      className="onboarding-shell"
      role="dialog"
      aria-modal="true"
      aria-labelledby="onboarding-title"
    >
      <aside className="onboarding-intro">
        <div className="onboarding-brand">
          <span className="onboarding-brand-mark">
            <Sparkles size={22} />
          </span>
          <span>
            <b>DSS.</b>
            <small>VN30 INTELLIGENCE</small>
          </span>
        </div>
        <div>
          <p className="onboarding-kicker">Bắt đầu đúng với bạn</p>
          <h1>
            Ít nhiễu hơn.
            <br />
            Đúng trọng tâm hơn.
          </h1>
          <p>Ba lựa chọn ngắn giúp hệ thống hiểu cách hỗ trợ bạn.</p>
        </div>
        <small className="onboarding-note">
          Thông tin chỉ được lưu trên thiết bị này.
        </small>
      </aside>

      <section className="onboarding-panel">
        <div className="onboarding-topbar">
          <span>Câu {Math.min(step + 1, 3)} / 3</span>
          <div
            className="onboarding-progress"
            aria-label={`Tiến độ ${Math.min(step + 1, 3)} trên 3`}
          >
            {[0, 1, 2].map((index) => (
              <i key={index} className={index <= step ? "active" : ""} />
            ))}
          </div>
          {step < 3 && (
            <button type="button" onClick={onSkip}>
              Bỏ qua
            </button>
          )}
        </div>

        {key === "summary" ? (
          <div className="onboarding-content onboarding-result">
            <span className="result-icon">
              <Check size={28} />
            </span>
            <p className="onboarding-kicker">Đã xác định mục tiêu</p>
            <h2 id="onboarding-title">DSS đã hiểu điều bạn cần.</h2>
            <p>{profileSummary(profile)}</p>
            <dl className="profile-recap">
              <div>
                <dt>Kinh nghiệm</dt>
                <dd>{EXPERIENCE[profile.experience].title}</dd>
              </div>
              <div>
                <dt>Kiến thức</dt>
                <dd>{KNOWLEDGE[profile.knowledge].title}</dd>
              </div>
              <div>
                <dt>Mục tiêu</dt>
                <dd>{GOALS[profile.goal].title}</dd>
              </div>
            </dl>
          </div>
        ) : (
          <div className="onboarding-content">
            <p className="onboarding-kicker">Thiết lập trải nghiệm</p>
            <h2 id="onboarding-title">{prompts[key].title}</h2>
            <p>{prompts[key].body}</p>
            <div
              className={`onboarding-choices ${key === "goal" ? "goal-choices" : ""}`}
            >
              {Object.entries(options).map(([id, item]) => (
                <Choice
                  key={id}
                  id={id}
                  item={item}
                  selected={profile[key] === id}
                  onSelect={choose}
                />
              ))}
            </div>
          </div>
        )}

        <div className="onboarding-actions">
          {step > 0 && (
            <button
              type="button"
              className="onboarding-back"
              onClick={() => setStep((current) => current - 1)}
            >
              <ArrowLeft size={17} /> Quay lại
            </button>
          )}
          <button
            type="button"
            className="onboarding-next"
            disabled={!canContinue}
            onClick={() =>
              key === "summary"
                ? onComplete(profile)
                : setStep((current) => current + 1)
            }
          >
            {key === "summary" ? "Vào bảng phân tích" : "Tiếp tục"}{" "}
            <ArrowRight size={17} />
          </button>
        </div>
      </section>
    </div>
  );
}
