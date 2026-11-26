"""规则测试：干净数据不能误报，脏数据必须报出来。

误报比漏报更伤信任——一条天天喊狼来了的规则，两周后就没人看了。
所以「干净数据上必须 pass」是硬性要求，和「脏数据上必须 fail」同等重要。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rules import (  # noqa: E402
    check_completeness,
    check_device_consistency,
    check_duplicate,
    check_funnel_order,
    check_timeliness,
    check_validity,
)


def make_clean(n_days: int = 15, sessions_per_day: int = 1500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2024-06-01")
    for d in range(n_days):
        for i in range(sessions_per_day):
            sid = f"s{d}_{i}"
            uid = int(rng.integers(1, 3000))
            dev = f"d{uid}"
            platform = ["ios", "android", "web"][int(rng.integers(0, 3))]
            t0 = start + pd.Timedelta(days=d, hours=int(rng.integers(6, 22)))
            rows.append(dict(event_id=f"{sid}v", user_id=uid, device_id=dev, session_id=sid,
                             event="view", event_time=t0, server_time=t0 + pd.Timedelta(seconds=2),
                             platform=platform, price=np.nan, utm_source="feed"))
            if rng.random() < 0.6:
                rows.append(dict(event_id=f"{sid}d", user_id=uid, device_id=dev, session_id=sid,
                                 event="detail", event_time=t0 + pd.Timedelta(seconds=30),
                                 server_time=t0 + pd.Timedelta(seconds=32),
                                 platform=platform, price=np.nan, utm_source="feed"))
            if rng.random() < 0.2:
                rows.append(dict(event_id=f"{sid}a", user_id=uid, device_id=dev, session_id=sid,
                                 event="add_to_cart", event_time=t0 + pd.Timedelta(seconds=60),
                                 server_time=t0 + pd.Timedelta(seconds=62),
                                 platform=platform, price=np.nan, utm_source="feed"))
            if rng.random() < 0.08:
                rows.append(dict(event_id=f"{sid}o", user_id=uid, device_id=dev, session_id=sid,
                                 event="order", event_time=t0 + pd.Timedelta(seconds=90),
                                 server_time=t0 + pd.Timedelta(seconds=92),
                                 platform=platform, price=199.0, utm_source="feed"))
    return pd.DataFrame(rows)


def test_clean_data_passes_all() -> None:
    df = make_clean()
    for res in [check_completeness(df), check_duplicate(df), check_timeliness(df),
                check_funnel_order(df), check_validity(df), check_device_consistency(df)]:
        assert res["status"] == "pass", f"{res['code']} 在干净数据上误报：{res['detail']}"


def test_missing_tracking_detected() -> None:
    df = make_clean(n_days=20)
    mask = (df["platform"] == "android") & (df["event"] == "add_to_cart") & (
        df["event_time"] >= pd.Timestamp("2024-06-11"))
    dirty = df[~mask]
    res = check_completeness(dirty)
    assert res["status"] == "fail", res
    assert "android" in res["detail"]


def test_duplicate_detected() -> None:
    df = make_clean()
    dup = df.sample(frac=0.03, random_state=0)
    res = check_duplicate(pd.concat([df, dup], ignore_index=True))
    assert res["status"] == "fail", res


def test_invalid_platform_detected() -> None:
    df = make_clean().copy()
    n_bad = int(len(df) * 0.02)   # 2% 越界，明显高于 0.5% 的阈值
    df.loc[df.index[:n_bad], "platform"] = "Android"
    res = check_validity(df)
    assert res["status"] == "fail", res


def test_inverted_funnel_detected() -> None:
    df = make_clean().copy()
    orders = df.index[df["event"] == "order"][:40]
    df.loc[orders, "event_time"] = df.loc[orders, "event_time"] - pd.Timedelta(hours=3)
    res = check_funnel_order(df)
    assert res["status"] == "fail", res


def test_device_reuse_detected() -> None:
    df = make_clean().copy()
    sessions = df["session_id"].drop_duplicates().to_numpy()[:500]
    df.loc[df["session_id"].isin(sessions), "device_id"] = "d_shared"
    res = check_device_consistency(df)
    assert res["status"] == "fail", res


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"✓ {name}")
