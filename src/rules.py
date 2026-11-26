"""六类埋点质检规则。

每条规则都返回统一结构，方便直接拼成质量日报：
    code / name / status / metric / value / detail / affected_rows

设计原则：规则必须「可解释 + 可定位」。只说「今天数据异常」没有用，
运营要的是「哪个端、哪个事件、从哪天开始、影响多少行」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALLOWED_PLATFORMS = {"ios", "android", "web"}
FUNNEL = ["view", "detail", "add_to_cart", "order"]


def _result(code: str, name: str, status: str, metric: str, value, detail: str,
            affected_rows: int = 0) -> dict:
    return {
        "code": code, "name": name, "status": status, "metric": metric,
        "value": value, "detail": detail, "affected_rows": int(affected_rows),
    }


def check_completeness(df: pd.DataFrame, drop_threshold: float = 0.4,
                       min_volume: int = 200) -> dict:
    """R1 完整性：某个端某个环节的埋点整段缺失。

    做法：算「每个平台每天的加购事件数 / 商详事件数」，看是否出现断崖。

    基线用 90 分位而不是中位数——这是个踩过的坑：漏埋一旦持续超过半个月，
    坏值就成了「多数」，中位数会被拉到 0，MAD 也变成 0，稳健 z 反而判不出异常。
    90 分位代表「健康水平」，不受坏值天数多少影响。

    另外要求平台日商详量 ≥ 200 才参与判定：窗口最后一天常常只剩几十个会话，
    这种小样本的随机波动会被误判成埋点挂了，误报比漏报更伤信任。
    """
    daily = (
        df[df["event"].isin(["detail", "add_to_cart"])]
        .assign(day=df["event_time"].dt.date)
        .pivot_table(index=["day", "platform"], columns="event",
                     values="event_id", aggfunc="count")
        .fillna(0.0)
    )
    if "add_to_cart" not in daily:
        return _result("R1", "完整性", "fail", "加购/商详比", 0.0, "加购事件完全缺失", len(df))
    daily["ratio"] = daily["add_to_cart"] / daily["detail"].replace(0, np.nan)

    anomalies = []
    for platform, sub in daily.groupby(level="platform"):
        sub = sub[sub["detail"] >= min_volume]
        if len(sub) < 7:
            continue
        ratio = sub["ratio"].fillna(0.0)
        baseline = float(ratio.quantile(0.90))
        if baseline <= 0:
            continue
        bad = sub[ratio < baseline * (1 - drop_threshold)]
        for day, row in bad.iterrows():
            anomalies.append({
                "day": str(day[0]), "platform": platform,
                "ratio": float(row["ratio"]), "baseline": baseline,
                "drop": 1 - float(row["ratio"]) / baseline,
            })

    if not anomalies:
        return _result("R1", "完整性", "pass", "加购/商详比", None,
                       "各端加购率未见断崖式下跌")

    worst = max(anomalies, key=lambda a: a["drop"])
    first_day = min(pd.Timestamp(a["day"]).date() for a in anomalies)
    affected = int(
        daily.loc[(daily.index.get_level_values("day") >= first_day)
                  & (daily.index.get_level_values("platform") == worst["platform"]),
                  "detail"].sum()
    )
    return _result(
        "R1", "完整性", "fail", "加购/商详比",
        round(worst["ratio"], 4),
        f"{worst['platform']} 端自 {first_day} 起加购/商详比从 {worst['baseline']:.3f} "
        f"跌到 {worst['ratio']:.3f}（跌幅 {worst['drop']:.0%}），"
        f"共 {len(anomalies)} 个平台日触发；同期商详量 {affected:,} 行",
        affected,
    )


def check_duplicate(df: pd.DataFrame, threshold: float = 0.01) -> dict:
    """R2 唯一性：event_id 重复上报（SDK 重试最常见）。"""
    total = len(df)
    dup_rows = int(total - df["event_id"].nunique())
    rate = dup_rows / total
    status = "fail" if rate > threshold else "pass"
    return _result("R2", "唯一性", status, "重复率", round(rate, 4),
                   f"{dup_rows:,} 行 event_id 重复，去重后 {df['event_id'].nunique():,} 行",
                   dup_rows)


def check_timeliness(df: pd.DataFrame, p95_threshold_s: float = 600.0) -> dict:
    """R3 及时性：上报延迟与整点断流。

    延迟 = server_time - event_time。注意客户端时钟不准也会造成假延迟，
    所以这里只报「整小时 p95 超标」，不逐行判定。
    """
    df = df.copy()
    df["latency_s"] = (df["server_time"] - df["event_time"]).dt.total_seconds()
    hourly = df.set_index("event_time").resample("1h").agg(
        n=("event_id", "count"), p95=("latency_s", lambda s: s.quantile(0.95))
    )
    late_hours = hourly[hourly["p95"] > p95_threshold_s]

    # 断点判定要排除「本来就没人来的时段」：凌晨 3 点没有日志是正常的，
    # 一律按 n==0 判会把每天的低谷都报成故障。用同一小时的历史中位数做期望值。
    hour_of_day = hourly.index.hour
    expected = hourly["n"].groupby(hour_of_day).transform("median")
    gap_hours = hourly[(hourly["n"] == 0) & (expected >= 5)]

    parts = []
    if len(late_hours):
        parts.append(f"{len(late_hours)} 个整点 p95 延迟超 {p95_threshold_s:.0f}s"
                     f"（最高 {late_hours['p95'].max():.0f}s，"
                     f"峰值时段 {late_hours['p95'].idxmax():%m-%d %H:%M}）")
    if len(gap_hours):
        parts.append(f"{len(gap_hours)} 个整点无任何日志"
                     f"（{gap_hours.index[0]:%m-%d %H:%M} 起连续 {len(gap_hours)} 小时）")
    status = "fail" if (len(late_hours) or len(gap_hours)) else "pass"
    affected = int(late_hours["n"].sum() + 0) if len(late_hours) else 0
    return _result("R3", "及时性", status, "整点 p95 延迟",
                   round(float(hourly["p95"].quantile(0.95)), 1),
                   "；".join(parts) if parts else "延迟与到达率正常", affected)


def check_funnel_order(df: pd.DataFrame) -> dict:
    """R4 一致性：漏斗顺序倒挂（下单时间早于商详）。"""
    piv = df[df["event"].isin(["detail", "order"])].pivot_table(
        index="session_id", columns="event", values="event_time", aggfunc="min"
    )
    both = piv.dropna()
    if both.empty:
        return _result("R4", "一致性", "pass", "倒挂会话占比", 0.0, "无足够样本")
    inverted = both["order"] < both["detail"]
    rate = float(inverted.mean())
    status = "fail" if rate > 0.005 else "pass"
    return _result("R4", "一致性", status, "倒挂会话占比", round(rate, 4),
                   f"{int(inverted.sum()):,} 个会话的下单时间早于商详时间，"
                   f"疑似客户端时钟未同步（占有效会话 {rate:.1%}）",
                   int(inverted.sum()))


def check_validity(df: pd.DataFrame) -> dict:
    """R5 有效性：枚举越界、金额非正、设备号缺失。"""
    bad_platform = ~df["platform"].isin(ALLOWED_PLATFORMS)
    bad_price = (df["event"] == "order") & (df["price"].fillna(0) <= 0)
    missing_device = df["device_id"].isna() | (df["device_id"].astype(str) == "")
    total_bad = int(bad_platform.sum() + bad_price.sum())
    rate = total_bad / len(df)
    status = "fail" if rate > 0.005 else "pass"
    return _result(
        "R5", "有效性", status, "非法值占比", round(rate, 4),
        f"platform 非法 {int(bad_platform.sum()):,} 行"
        f"（如 {', '.join(sorted(df.loc[bad_platform, 'platform'].unique())[:3])}）；"
        f"订单金额非正 {int(bad_price.sum()):,} 行；device_id 缺失 {missing_device.mean():.1%}",
        total_bad,
    )


def check_device_consistency(df: pd.DataFrame, threshold: float = 0.01) -> dict:
    """R6 设备一致性：一个 device_id 对应多个 user_id（模拟器、家庭共用设备）。"""
    d = df[df["device_id"].astype(str) != ""]
    per_device = d.groupby("device_id")["user_id"].nunique()
    reused = per_device[per_device > 1]
    dev_rate = len(reused) / max(len(per_device), 1)
    touched = d[d["device_id"].isin(reused.index)]
    affected = int(touched.shape[0])
    affected_sessions = int(touched["session_id"].nunique())
    session_rate = affected_sessions / max(d["session_id"].nunique(), 1)
    status = "fail" if session_rate > threshold else "pass"
    return _result("R6", "设备一致性", status, "受影响会话占比", round(session_rate, 4),
                   f"{len(reused):,} 个设备号关联多个用户（占设备数 {dev_rate:.1%}），"
                   f"波及 {affected:,} 行事件、{affected_sessions:,} 个会话（{session_rate:.1%}）；"
                   f"按设备去重会把 UV 算少",
                   affected)


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        check_completeness(df),
        check_duplicate(df),
        check_timeliness(df),
        check_funnel_order(df),
        check_validity(df),
        check_device_consistency(df),
    ]
    return pd.DataFrame(checks)


def detection_quality(df: pd.DataFrame) -> pd.DataFrame:
    """对着注入清单算检出质量。

    关键点：不同缺陷的「可检测粒度」不一样，不能一律按行算。
    - 重复上报：两条记录的 event_id 相同，谁才是副本无从判断，
      只能在 event_id 维度判定，行级 P/R 没有意义。
    - 顺序倒挂：天然是会话级现象。
    - 设备复用：天然是设备级现象。
    - 字段非法：行级可直接判定。
    """

    def score(name: str, granularity: str, truth: set, pred: set) -> dict:
        tp = len(truth & pred)
        return {
            "defect": name, "granularity": granularity,
            "injected": len(truth), "detected": len(pred),
            "true_positive": tp, "false_positive": len(pred - truth),
            "false_negative": len(truth - pred),
            "precision": tp / max(len(pred), 1),
            "recall": tp / max(len(truth), 1),
        }

    rows = []

    # D2 重复：event_id 级
    ids_injected = set(df.loc[df["flag_dup"] == 1, "event_id"])
    ids_detected = set(df.loc[df.duplicated("event_id", keep=False), "event_id"])
    rows.append(score("D2 重复上报", "event_id", ids_injected, ids_detected))

    # D4 顺序倒挂：会话级
    inv_sessions = set(df.loc[df["flag_inversion"] == 1, "session_id"])
    piv = df[df["event"].isin(["detail", "order"])].pivot_table(
        index="session_id", columns="event", values="event_time", aggfunc="min"
    ).dropna()
    detected = set(piv[piv["order"] < piv["detail"]].index)
    rows.append(score("D4 顺序倒挂", "session_id", inv_sessions, detected))

    # D5 字段非法：行级
    idx_truth = set(df.index[df["flag_bad_field"] == 1])
    idx_pred = set(df.index[(~df["platform"].isin(ALLOWED_PLATFORMS))
                            | ((df["event"] == "order") & (df["price"].fillna(0) <= 0))])
    rows.append(score("D5 字段异常", "row", idx_truth, idx_pred))

    # D6 设备复用：设备级
    dev_truth = set(df.loc[df["flag_reuse"] == 1, "device_id"])
    counts = df[df["device_id"].astype(str) != ""].groupby("device_id")["user_id"].nunique()
    dev_pred = set(counts[counts > 1].index)
    rows.append(score("D6 设备复用", "device_id", dev_truth, dev_pred))

    return pd.DataFrame(rows)
