"""影响面估算：把「数据有问题」翻译成「哪个指标错了多少」。

只报异常不报影响，业务方不会改。这里对三个核心指标做观测值 / 校正值对照：
    1. 加购率（加购会话 / 商详会话）——受漏埋直接影响
    2. 支付订单的 GMV——受金额非法直接影响
    3. 事件量 PV——受重复上报直接影响

校正方法都写在函数里：能外推的外推（用缺陷发生前的自身比例），
不能外推的用同平台同天中位数兜底，并标注这是估计而非测量。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFECT_DAY = "2024-06-11"


def add_cart_rate_bias(df: pd.DataFrame, platform: str = "android") -> dict:
    """漏埋对加购率的影响：用该端缺陷发生前的自身比例外推。"""
    d = df[df["platform"] == platform].copy()
    d["day"] = d["event_time"].dt.date
    cutoff = pd.Timestamp(DEFECT_DAY).date()

    pre = d[d["day"] < cutoff]
    post = d[d["day"] >= cutoff]

    def rate(sub: pd.DataFrame) -> tuple[float, int, int]:
        s = sub.groupby("session_id")["event"].agg(lambda x: set(x))
        detail = s.map(lambda v: "detail" in v).sum()
        cart = s.map(lambda v: "add_to_cart" in v).sum()
        return cart / max(detail, 1), int(cart), int(detail)

    pre_rate, _, _ = rate(pre)
    obs_rate, obs_cart, post_detail = rate(post)
    expected_cart = int(round(pre_rate * post_detail))

    return {
        "platform": platform,
        "pre_rate": pre_rate,
        "observed_rate": obs_rate,
        "expected_rate": pre_rate,
        "post_detail_sessions": post_detail,
        "observed_cart_sessions": obs_cart,
        "expected_cart_sessions": expected_cart,
        "missing_cart_sessions": max(expected_cart - obs_cart, 0),
        "relative_bias": obs_rate / pre_rate - 1 if pre_rate > 0 else float("nan"),
    }


def overall_add_cart_rate(df: pd.DataFrame, platform: str = "android") -> dict:
    """全站加购率：观测值 vs 把缺失部分补回来的校正值。"""
    s = df.groupby("session_id").agg(
        events=("event", lambda x: set(x)), platform=("platform", "first"),
        day=("event_time", "min"),
    )
    s["has_detail"] = s["events"].map(lambda v: "detail" in v)
    s["has_cart"] = s["events"].map(lambda v: "add_to_cart" in v)
    obs = s["has_cart"].sum() / s["has_detail"].sum()

    bias = add_cart_rate_bias(df, platform)
    corrected_cart = s["has_cart"].sum() + bias["missing_cart_sessions"]
    corrected = corrected_cart / s["has_detail"].sum()

    return {
        "observed": float(obs),
        "corrected": float(corrected),
        "absolute_gap_pp": float((corrected - obs) * 100),
        "relative_gap": float(corrected / obs - 1),
        "missing_cart_sessions": bias["missing_cart_sessions"],
    }


def gmv_bias(df: pd.DataFrame) -> dict:
    """金额非法对 GMV 的影响：用同平台订单金额中位数回填。"""
    orders = df[df["event"] == "order"].copy()
    orders["price"] = orders["price"].astype(float)
    bad = orders["price"].isna() | (orders["price"] <= 0)
    valid = ~bad

    median_by_platform = orders[valid].groupby("platform")["price"].median()
    fill = orders.loc[bad, "platform"].map(median_by_platform).fillna(
        orders.loc[valid, "price"].median()
    )
    observed = float(orders.loc[valid, "price"].sum())
    corrected = float(observed + fill.sum())

    return {
        "orders_total": int(len(orders)),
        "orders_invalid_price": int(bad.sum()),
        "invalid_share": float(bad.mean()),
        "gmv_observed": observed,
        "gmv_corrected": corrected,
        "gmv_gap": corrected - observed,
        "relative_gap": corrected / observed - 1 if observed else float("nan"),
    }


def pv_bias(df: pd.DataFrame) -> dict:
    """重复上报对 PV 的影响：去重前后的事件量差。"""
    total = len(df)
    dedup = df.drop_duplicates("event_id")
    return {
        "pv_observed": total,
        "pv_deduplicated": int(len(dedup)),
        "inflation": total / len(dedup) - 1,
    }


def build_report(df: pd.DataFrame) -> pd.DataFrame:
    acr = overall_add_cart_rate(df)
    gmv = gmv_bias(df)
    pv = pv_bias(df)
    return pd.DataFrame([
        {"metric": "全站加购率", "observed": acr["observed"], "corrected": acr["corrected"],
         "gap": f"{acr['absolute_gap_pp']:+.2f} pp", "direction": "低估",
         "note": f"补回 {acr['missing_cart_sessions']:,} 个缺失加购会话"},
        {"metric": "GMV（元）", "observed": gmv["gmv_observed"], "corrected": gmv["gmv_corrected"],
         "gap": f"{gmv['relative_gap']:+.1%}", "direction": "低估",
         "note": f"{gmv['orders_invalid_price']:,} 笔订单金额非法，按平台中位数回填"},
        {"metric": "事件量 PV", "observed": pv["pv_observed"], "corrected": pv["pv_deduplicated"],
         "gap": f"{-pv['inflation']:+.1%}", "direction": "高估",
         "note": "event_id 重复上报，去重后为真实量"},
    ])
