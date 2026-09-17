"""Thành phần VN30 theo từng kỳ, dựng từ reference/vn30_changes.csv.

File ghi rổ gốc ngày 2020-08-03 (`base`) và mọi lần thêm/loại (`add`/`remove`)
kèm nguồn công bố. Dùng để backtest không bị survivorship bias: tại mỗi ngày
chỉ giao dịch, xếp hạng và train trên các mã đang thuộc rổ ngày đó.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config

CHANGES_PATH = os.path.join(config.BASE_DIR, "reference", "vn30_changes.csv")


def load_vn30_changes(path: str = CHANGES_PATH) -> pd.DataFrame:
    changes = pd.read_csv(path, parse_dates=["effective_date"])
    changes = changes[changes["action"] != "none"]
    return changes.sort_values("effective_date", kind="stable").reset_index(drop=True)


def vn30_symbols_ever(changes: pd.DataFrame) -> list[str]:
    return sorted(changes["symbol"].unique())


def vn30_members(changes: pd.DataFrame, day) -> set[str]:
    """Rổ VN30 có hiệu lực tại `day` (trước ngày gốc thì dùng rổ gốc)."""
    members: set[str] = set()
    for row in changes.itertuples():
        if row.action == "base":
            members.add(row.symbol)
        elif row.effective_date <= pd.Timestamp(day):
            (members.add if row.action == "add" else members.discard)(row.symbol)
    return members


def membership_mask(changes: pd.DataFrame, symbol: str, times: pd.Series) -> np.ndarray:
    """True tại các ngày `symbol` thuộc VN30.

    Ngày trước rổ gốc dùng rổ gốc (xấp xỉ, vì không có dữ liệu thành phần cũ hơn).
    """
    events = changes[changes["symbol"] == symbol]
    if events.empty:
        return np.zeros(len(times), dtype=bool)
    in_basket = events["action"].isin(["base", "add"]).to_numpy()
    last_event = np.searchsorted(events["effective_date"].to_numpy(), times.to_numpy(), side="right") - 1
    before_first = in_basket[0] and events["action"].iloc[0] == "base"
    return np.where(last_event >= 0, in_basket[np.maximum(last_event, 0)], before_first)
