"""埋点质量图表：事件量、加购率断崖、整点延迟。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linestyle"] = "--"

BLUE, ORANGE, TEAL, RED, MUTED = "#2f6feb", "#e8833a", "#2f9e8f", "#d64545", "#8a94a6"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def volume_by_platform(df: pd.DataFrame, path: Path) -> None:
    daily = (
        df.assign(day=df["event_time"].dt.floor("D"))
        .groupby(["day", "platform"])["event_id"].count().unstack("platform").fillna(0)
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    colors = {"ios": BLUE, "android": ORANGE, "web": TEAL}
    for platform in daily.columns:
        ax.plot(daily.index, daily[platform], lw=1.6, color=colors.get(platform, MUTED),
                label=platform)
    ax.axvline(pd.Timestamp("2024-06-11"), color=RED, ls="--", lw=1.2)
    ax.annotate("6/11 Android 加购埋点缺失", xy=(pd.Timestamp("2024-06-11"), daily.max().max() * 0.95),
                xytext=(6, -4), textcoords="offset points", fontsize=8.5, color=RED)
    ax.set_ylabel("事件量（行）")
    ax.set_xlabel("日期")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower left")
    _save(fig, path)


def cart_ratio(df: pd.DataFrame, path: Path) -> None:
    daily = (
        df[df["event"].isin(["detail", "add_to_cart"])]
        .assign(day=df["event_time"].dt.floor("D"))
        .pivot_table(index=["day", "platform"], columns="event",
                     values="event_id", aggfunc="count").fillna(0)
    )
    daily["ratio"] = daily["add_to_cart"] / daily["detail"].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    colors = {"ios": BLUE, "android": ORANGE, "web": TEAL}
    for platform, sub in daily.groupby(level="platform"):
        s = sub.droplevel("platform")["ratio"]
        ax.plot(s.index, s.values, lw=1.5, marker="o", ms=2.5,
                color=colors.get(platform, MUTED), label=platform)
    ax.axvline(pd.Timestamp("2024-06-11"), color=RED, ls="--", lw=1.2)
    ax.set_ylabel("加购事件 / 商详事件")
    ax.set_xlabel("日期")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9, ncol=3)
    _save(fig, path)


def hourly_latency(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    df["latency_min"] = (df["server_time"] - df["event_time"]).dt.total_seconds() / 60
    hourly = df.set_index("event_time").resample("1h").agg(
        n=("event_id", "count"),
        p95=("latency_min", lambda s: s.quantile(0.95)),
    ).loc["2024-06-18"]
    fig, ax1 = plt.subplots(figsize=(7.4, 3.6))
    ax1.bar(hourly.index.hour, hourly["n"], color=MUTED, alpha=0.45, width=0.7,
            label="事件量")
    ax1.set_xlabel("6/18 当日小时")
    ax1.set_ylabel("事件量（行）", color=MUTED)
    ax2 = ax1.twinx()
    ax2.plot(hourly.index.hour, hourly["p95"], color=RED, lw=1.8, marker="o", ms=3,
             label="p95 延迟（分钟）")
    ax2.set_ylabel("p95 上报延迟（分钟）", color=RED)
    ax2.spines["right"].set_visible(True)
    ax2.axhline(10, color=ORANGE, ls="--", lw=1)
    ax2.annotate("延迟 180 分钟\n（20:00-22:00）", xy=(21, 180), xytext=(-96, -26),
                 textcoords="offset points", fontsize=8.5, color=RED)
    ax1.annotate("04:00 无日志", xy=(4, 0), xytext=(-14, 26), textcoords="offset points",
                 fontsize=8.5, color=BLUE)
    _save(fig, path)


def bias_bars(report: pd.DataFrame, path: Path) -> None:
    """三个核心指标的相对偏差（把不同量纲统一成百分比）。"""
    d = report[["metric", "observed", "corrected"]].copy()
    d["rel"] = d["corrected"] / d["observed"] - 1
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    colors = [RED if v < 0 else BLUE for v in d["rel"]]
    ax.barh(d["metric"], d["rel"] * 100, color=colors, height=0.5)
    for i, v in enumerate(d["rel"] * 100):
        ax.text(v + (1.2 if v >= 0 else -1.2), i, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_xlabel("观测值相对校正值的偏差（%）")
    ax.set_xlim(-12, 45)
    _save(fig, path)
