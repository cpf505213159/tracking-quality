"""埋点日志生成器：造一份「有已知缺陷」的日志，用于验证质检规则。

为什么用生成器而不是公开数据集
------------------------------
公开数据集里没有「埋点上报」这种带缺陷标记的日志。要证明质检规则有效，
必须知道哪些行是坏的——所以这里先造一份干净的日志，再按剧本注入缺陷，
并把注入清单存成 ground_truth.json。规则引擎跑完以后，检出率、误报率、
影响面偏差都可以对着真值算，不用自说自话。

注入的六类缺陷（都对应真实踩过的坑）
----------------------------------
D1 漏埋      Android 端 6/11 起 add_to_cart 事件整段丢失（发版漏传参数）
D2 重复上报  5% 事件被 SDK 重试机制上报两次（event_id 相同）
D3 延迟/断点 6/18 20:00-22:00 上报延迟 3 小时；6/18 04:00 整点无日志
D4 顺序倒挂  3% 会话的 order 事件时间早于 detail（时钟不同步）
D5 字段异常  2% 事件 platform 写入非法值；1% 订单 price ≤ 0
D6 设备复用  4% 会话的 device_id 与其他用户共用（模拟器/家庭共享设备）

运行：python -m src.generate
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20240630
N_USERS = 8000
N_SESSIONS = 60000
START = "2024-06-01"
N_DAYS = 30

EVENTS = ["view", "detail", "add_cart", "order"]
# 会话内每推进一步的转化概率（真实电商大致量级）
STEP_PROB = {"view": 1.0, "detail": 0.58, "add_cart": 0.34, "order": 0.42}
PLATFORMS = ["ios", "android", "web"]
PLATFORM_P = [0.34, 0.41, 0.25]

# 缺陷剧本
D1_START_DAY = 10          # 6/11 起（0-based）
D1_PLATFORM = "android"
D1_EVENT = "add_to_cart"
D2_DUP_RATE = 0.05
D3_LATE_DAY = 17           # 6/18
D3_LATE_HOURS = (20, 22)
D3_LATE_SECONDS = 3 * 3600
D3_GAP_HOUR = 4
D4_INVERSION_RATE = 0.03
D5_BAD_PLATFORM_RATE = 0.02
D5_BAD_PRICE_RATE = 0.01
D6_REUSE_RATE = 0.01


def _build_clean(rng: np.random.Generator) -> pd.DataFrame:
    user_ids = np.arange(1, N_USERS + 1)
    rows: list[dict] = []

    # 用户活跃度：长尾分布，少数重度用户撑起多数会话
    user_weight = rng.pareto(1.6, N_USERS) + 1
    user_weight /= user_weight.sum()

    session_user = rng.choice(user_ids, size=N_SESSIONS, p=user_weight)
    day_choice = rng.choice(N_DAYS, size=N_SESSIONS, p=_day_weights())
    # 会话开始时间：日内双峰（午休、晚间）
    hour_choice = rng.choice(24, size=N_SESSIONS, p=_hour_weights())
    minute = rng.integers(0, 60, N_SESSIONS)
    second = rng.integers(0, 60, N_SESSIONS)

    for i in range(N_SESSIONS):
        uid = int(session_user[i])
        day = int(day_choice[i])
        base = pd.Timestamp(START) + pd.Timedelta(days=day, hours=int(hour_choice[i]),
                                                  minutes=int(minute[i]), seconds=int(second[i]))
        platform = str(rng.choice(PLATFORMS, p=PLATFORM_P))
        device = f"d{uid:06d}" if rng.random() > 0.08 else ""
        session_id = f"s{i:07d}"
        # 一个会话最多走到 order，走不到就停
        depth = 1
        for nxt in EVENTS[1:]:
            if rng.random() < STEP_PROB[nxt]:
                depth += 1
            else:
                break
        ts = base
        # 每个会话先看 1~4 个页面，再决定是否进商详
        for v in range(int(rng.integers(1, 5))):
            rows.append({
                "event_id": f"e{i:07d}v{v}", "user_id": uid, "device_id": device,
                "session_id": session_id, "event": "view", "event_time": ts,
                "server_time": ts + pd.Timedelta(seconds=float(rng.gamma(2.0, 1.4))),
                "platform": platform, "price": np.nan,
                "utm_source": str(rng.choice(["search", "feed", "direct", "affiliate"],
                                             p=[0.32, 0.41, 0.19, 0.08])),
            })
            ts = ts + pd.Timedelta(seconds=float(rng.integers(20, 180)))
        for step in range(1, depth):
            ev = EVENTS[step]
            rows.append(
                {
                    "event_id": f"e{i:07d}{step}",
                    "user_id": uid,
                    "device_id": device,
                    "session_id": session_id,
                    "event": ev,
                    "event_time": ts,
                    "server_time": ts + pd.Timedelta(seconds=float(rng.gamma(2.0, 1.4))),
                    "platform": platform,
                    "price": float(rng.lognormal(4.6, 0.6)) if ev == "order" else np.nan,
                    "utm_source": str(rng.choice(["search", "feed", "direct", "affiliate"],
                                                 p=[0.32, 0.41, 0.19, 0.08])),
                }
            )
            ts = ts + pd.Timedelta(seconds=float(rng.integers(8, 240)))

    return pd.DataFrame(rows)


def _day_weights() -> np.ndarray:
    w = np.ones(N_DAYS)
    w[16:20] *= 1.9   # 618 大促
    w[9] *= 1.1
    return w / w.sum()


def _hour_weights() -> np.ndarray:
    w = np.array([0.6, 0.3, 0.2, 0.15, 0.15, 0.3, 0.8, 1.6, 2.4, 2.8, 3.0, 3.2,
                  3.4, 2.9, 2.9, 3.0, 3.2, 3.4, 3.8, 4.4, 5.0, 5.2, 4.4, 2.8])
    return w / w.sum()


def _inject(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["server_time"] = pd.to_datetime(df["server_time"])
    df["flag_dup"] = 0
    df["flag_late"] = 0
    df["flag_inversion"] = 0
    df["flag_bad_field"] = 0
    df["flag_reuse"] = 0

    start = pd.Timestamp(START)

    # 内部规范值 add_cart 统一成埋点名 add_to_cart（真实日志里两种写法都出现过）
    df["event"] = df["event"].replace({"add_cart": "add_to_cart"})

    # D1 漏埋：整段删除，检测端只能靠「比例异常」发现
    drop_mask = (
        (df["platform"] == D1_PLATFORM)
        & (df["event"] == D1_EVENT)
        & ((df["event_time"] - start).dt.days >= D1_START_DAY)
    )
    dropped = int(drop_mask.sum())
    df = df[~drop_mask].copy()

    # D3 延迟：服务器时间整段后移
    late_mask = (
        (df["event_time"] - start).dt.days == D3_LATE_DAY
    ) & df["event_time"].dt.hour.between(*D3_LATE_HOURS)
    df.loc[late_mask, "server_time"] = df.loc[late_mask, "event_time"] + pd.Timedelta(
        seconds=D3_LATE_SECONDS
    )
    df.loc[late_mask, "flag_late"] = 1
    late_rows = int(late_mask.sum())

    # D3 断点：整点日志消失
    gap_mask = ((df["event_time"] - start).dt.days == D3_LATE_DAY) & (
        df["event_time"].dt.hour == D3_GAP_HOUR
    )
    gap_rows = int(gap_mask.sum())
    df = df[~gap_mask].copy()

    # D4 顺序倒挂：把 order 的时间戳提前到 detail 之前
    order_idx = df.index[df["event"] == "order"]
    n_inv = int(len(order_idx) * D4_INVERSION_RATE)
    inv_idx = rng.choice(order_idx, size=n_inv, replace=False)
    df.loc[inv_idx, "event_time"] = df.loc[inv_idx, "event_time"] - pd.Timedelta(hours=2)
    df.loc[inv_idx, "flag_inversion"] = 1

    # D5 字段异常
    n_bad_platform = int(len(df) * D5_BAD_PLATFORM_RATE)
    bad_p_idx = rng.choice(df.index, size=n_bad_platform, replace=False)
    df.loc[bad_p_idx, "platform"] = rng.choice(["Android", "iOS ", "web3"], size=n_bad_platform)
    df.loc[bad_p_idx, "flag_bad_field"] = 1

    order_rows = df.index[df["event"] == "order"]
    n_bad_price = int(len(order_rows) * D5_BAD_PRICE_RATE)
    bad_price_idx = rng.choice(order_rows, size=n_bad_price, replace=False)
    df.loc[bad_price_idx, "price"] = rng.choice([-1.0, 0.0], size=n_bad_price)
    df.loc[bad_price_idx, "flag_bad_field"] = 1

    # D6 设备复用：把部分会话的 device_id 换成别人的
    session_ids = df["session_id"].drop_duplicates().to_numpy()
    n_reuse = int(len(session_ids) * D6_REUSE_RATE)
    reuse_sessions = rng.choice(session_ids, size=n_reuse, replace=False)
    # 目标设备从用户里均匀抽，而不是从设备池里抽：
    # 设备池里热门设备权重极高，抽到它会把「1% 会话受影响」放大成 90%
    target_users = rng.choice(df["user_id"].unique(), size=len(reuse_sessions))
    mapping = {s: f"d{u:06d}" for s, u in zip(reuse_sessions, target_users)}
    mask = df["session_id"].isin(reuse_sessions)
    df.loc[mask, "device_id"] = df.loc[mask, "session_id"].map(mapping)
    df.loc[mask, "flag_reuse"] = 1

    # D2 重复上报：同一 event_id 再写一遍
    dup = df.sample(frac=D2_DUP_RATE, random_state=SEED).copy()
    dup["server_time"] = dup["server_time"] + pd.Timedelta(seconds=2)
    dup["flag_dup"] = 1
    df["flag_dup"] = 0
    df = pd.concat([df, dup], ignore_index=True)

    df = df.sort_values("event_time").reset_index(drop=True)

    truth = {
        "seed": SEED,
        "n_rows_final": int(len(df)),
        "windows": {"start": START, "days": N_DAYS},
        "defects": {
            "D1_missing_tracking": {
                "platform": D1_PLATFORM, "event": D1_EVENT,
                "from_day": D1_START_DAY, "rows_removed": dropped,
                "description": "Android 端 add_to_cart 自 6/11 起整段缺失",
            },
            "D2_duplicate": {"rate": D2_DUP_RATE, "rows_duplicated": int(len(dup))},
            "D3_late": {"day": D3_LATE_DAY, "hours": list(D3_LATE_HOURS),
                        "delay_seconds": D3_LATE_SECONDS, "rows": late_rows},
            "D3_gap": {"day": D3_LATE_DAY, "hour": D3_GAP_HOUR, "rows_removed": gap_rows},
            "D4_inversion": {"rate": D4_INVERSION_RATE, "rows": int(n_inv)},
            "D5_bad_field": {"platform_rate": D5_BAD_PLATFORM_RATE,
                             "price_rate": D5_BAD_PRICE_RATE,
                             "rows": int(n_bad_platform + n_bad_price)},
            "D6_device_reuse": {"rate": D6_REUSE_RATE, "sessions": int(n_reuse)},
        },
    }
    return df, truth


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    clean = _build_clean(rng)
    df, truth = _inject(clean, rng)
    df.to_csv(out / "events.csv", index=False, encoding="utf-8")
    (out / "ground_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"生成事件 {len(df):,} 行｜用户 {df['user_id'].nunique():,}｜"
          f"会话 {df['session_id'].nunique():,}｜窗口 {START} 起 {N_DAYS} 天")
    print("注入缺陷：" + json.dumps(truth["defects"], ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
