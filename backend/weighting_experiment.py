"""Đối chứng 2×2 balancing/recency trên cùng pooled excess T+20 và luật giao dịch."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd

import config
from backtest_runner import load_universe
from src.backtest.backtester import daily_rank_ic, equal_weight_return, index_return, score_history, simulate_portfolio
from src.features import FEATURE_COLUMNS
from src.models.metrics import classification_metrics
from src.models.ml_models import LABEL_TO_CLASS, TrainingWeights, model_config
from src.pipeline import load_market_index

HALF_LIFE_DAYS = 365.25 * 2
VARIANTS = {
    "balanced_uniform": TrainingWeights(),
    "unbalanced_uniform": TrainingWeights(class_balance=None),
    "balanced_decay_2y": TrainingWeights(half_life_days=HALF_LIFE_DAYS),
    "unbalanced_decay_2y": TrainingWeights(class_balance=None, half_life_days=HALF_LIFE_DAYS),
}


def prediction_metrics(scores: pd.DataFrame) -> dict:
    eligible = scores[scores["in_universe"] & scores["ml_prediction"].notna()]
    labelled = eligible.dropna(subset=["label"])
    metrics = classification_metrics(labelled["label"].map(LABEL_TO_CLASS).astype(int),
                                     labelled["ml_prediction"].astype(int))
    metrics.pop("confusion_matrix")
    buy = labelled["ml_score"] >= config.SCORE_BUY
    actual_buy = labelled["label"] == 1
    return {
        **{f"class_{key}": value for key, value in metrics.items()},
        "labelled_rows": len(labelled),
        "labelled_end": labelled["time"].max(),
        "buy_signals": int((eligible["ml_score"] >= config.SCORE_BUY).sum()),
        "sell_signals": int((eligible["ml_score"] < config.SCORE_SELL).sum()),
        "labelled_buy_signals": int(buy.sum()),
        "signal_buy_precision": float(actual_buy[buy].mean()) if buy.any() else None,
        "signal_buy_recall": float(buy[actual_buy].mean()) if actual_buy.any() else None,
    }


def report_variant(scores: pd.DataFrame, args: argparse.Namespace, destination: Path) -> dict:
    portfolio = simulate_portfolio(scores, "ml_score", slots=args.slots, horizon=args.horizon)
    eligible = scores[scores["ml_prediction"].notna()]
    daily_ic = daily_rank_ic(eligible, "ml_score")
    yearly_ic = daily_ic.groupby(daily_ic.index.year).mean()
    year_end = portfolio["equity"].groupby(portfolio["equity"].index.year).last()
    yearly_return = year_end / year_end.shift(1, fill_value=1.0) - 1
    yearly = pd.DataFrame({"rank_ic": yearly_ic, "portfolio_return": yearly_return})
    yearly.index.name = "year"
    destination.mkdir()
    scores.to_csv(destination / "scores.csv", index=False)
    daily_ic.rename("rank_ic").to_csv(destination / "daily_rank_ic.csv")
    yearly.to_csv(destination / "by_year.csv")
    portfolio["equity"].rename("equity").to_csv(destination / "equity.csv")
    portfolio["trades"].to_csv(destination / "trades.csv", index=False)
    return {
        **prediction_metrics(scores),
        "rank_ic": float(daily_ic.mean()),
        "positive_ic_years": int((yearly_ic > 0).sum()),
        "ic_years": len(yearly_ic),
        **{key: portfolio[key] for key in ("portfolio_return", "num_trades", "win_rate",
                                          "avg_return", "sharpe", "max_drawdown", "exposure")},
    }


def save_manifest(featured: dict, args: argparse.Namespace, output: Path) -> None:
    data = pd.concat(featured, names=["symbol", "row"])
    sources = [Path(__file__), Path(config.__file__),
               Path(config.BASE_DIR) / "backtest_runner.py",
               Path(config.BASE_DIR) / "reference/vn30_changes.csv"]
    sources += sorted((Path(config.BASE_DIR) / "src").rglob("*.py"))
    coverage = []
    for symbol, frame in featured.items():
        complete = frame[FEATURE_COLUMNS + ["label"]].notna().all(axis=1)
        coverage.append({"symbol": symbol, "rows": len(frame), "start": frame.time.min(),
                         "end": frame.time.max(), "member_rows": int(frame.in_universe.sum()),
                         "trainable_member_rows": int((complete & frame.in_universe).sum())})
    pd.DataFrame(coverage).to_csv(output / "data_coverage.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_config": model_config(),
        "settings": vars(args),
        "variants": {name: asdict(weighting) for name, weighting in VARIANTS.items()},
        "data_start": str(data.time.min()), "data_end": str(data.time.max()),
        "symbols": sorted(featured),
        "featured_sha256": hashlib.sha256(pd.util.hash_pandas_object(data).values.tobytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(config.BASE_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "buy_threshold": config.SCORE_BUY, "sell_threshold": config.SCORE_SELL,
        "fee_per_side": config.BACKTEST_FEE_RATE, "sell_tax": config.BACKTEST_TAX_RATE,
        "slippage_per_side": config.BACKTEST_SLIPPAGE_RATE,
        "classification": "argmax after class-weight correction, only labelled historical members with valid features",
        "signal": "stock-days above entry threshold, not executed trades; precision tests the excess label",
        "limitations": ["Exploratory comparison on previously inspected history; no untouched holdout",
                        "Pre-2020-08-03 training membership approximated by the base basket",
                        "Labels remain close-to-close over N valid stock candles; execution remains next-open",
                        "Overlapping target windows; daily IC observations are not independent",
                        "Open positions marked at close without terminal liquidation costs; benchmark equal-weight is gross"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=72)
    parser.add_argument("--retrain-every", type=int, default=60)
    parser.add_argument("--slots", type=int, default=config.BACKTEST_PORTFOLIO_SLOTS)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if min(args.months, args.retrain_every, args.slots) <= 0:
        parser.error("months, retrain-every và slots phải > 0")
    args.universe, args.label_strategy, args.horizon = "history", "excess", 20
    args.symbols, args.limit = "", 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output or Path(config.REPORT_DIR) / f"weighting_ablation_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    index = load_market_index()
    if index.empty:
        raise ValueError("Thiếu VNINDEX trong cache")
    featured = load_universe(args, index)
    save_manifest(featured, args, output)
    rows = []
    for name, weighting in VARIANTS.items():
        started = perf_counter()
        print(f"\n{name}: {asdict(weighting)}", flush=True)
        scores = score_history(featured, args.months, args.retrain_every, pooled=True,
                               horizon=args.horizon, weighting=weighting)
        summary = report_variant(scores, args, output / name)
        rows.append({"variant": name, **summary, "seconds": perf_counter() - started,
                     "equal_weight_return": equal_weight_return(scores),
                     "vnindex_return": index_return(scores.time.drop_duplicates(), index)})
        pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
        print(f"{name}: return={summary['portfolio_return']:+.2%}, "
              f"IC={summary['rank_ic']:+.4f}, MDD={summary['max_drawdown']:.2%}, "
              f"trades={summary['num_trades']}", flush=True)
    print(f"\nReport: {output}", flush=True)


if __name__ == "__main__":
    main()
