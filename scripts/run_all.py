"""一条命令跑完 P3：生成带缺陷日志 → 六条规则 → 影响面 → 图表。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plots  # noqa: E402
from src.impact import (  # noqa: E402
    add_cart_rate_bias,
    build_report,
    gmv_bias,
    overall_add_cart_rate,
    pv_bias,
)  # noqa: E402
from src.rules import detection_quality, run_all  # noqa: E402


def main() -> None:
    out = ROOT / "outputs"
    fig_dir = ROOT / "reports" / "figures"
    events_path = out / "events.csv"

    if not events_path.exists():
        from src.generate import main as generate

        generate()

    df = pd.read_csv(events_path, parse_dates=["event_time", "server_time"])

    quality = run_all(df)
    quality.to_csv(out / "quality_report.csv", index=False, encoding="utf-8")

    detect = detection_quality(df)
    detect.to_csv(out / "detection_quality.csv", index=False, encoding="utf-8")

    impact = build_report(df)
    impact.to_csv(out / "impact.csv", index=False, encoding="utf-8")

    acr = overall_add_cart_rate(df)
    acr_platform = add_cart_rate_bias(df)
    gmv = gmv_bias(df)
    pv = pv_bias(df)

    plots.volume_by_platform(df, fig_dir / "volume_by_platform.png")
    plots.cart_ratio(df, fig_dir / "cart_ratio.png")
    plots.hourly_latency(df, fig_dir / "hourly_latency.png")
    plots.bias_bars(impact, fig_dir / "metric_bias.png")

    metrics = {
        "data": {
            "rows": int(len(df)),
            "users": int(df["user_id"].nunique()),
            "sessions": int(df["session_id"].nunique()),
            "days": int(df["event_time"].dt.date.nunique()),
            "start": str(df["event_time"].min()),
            "end": str(df["event_time"].max()),
        },
        "checks": quality[["code", "name", "status", "metric", "value", "affected_rows"]]
        .to_dict(orient="records"),
        "detection": detect.round(4).to_dict(orient="records"),
        "impact": {
            "add_cart_rate": {
                "observed": acr["observed"], "corrected": acr["corrected"],
                "bias_pp": (acr["observed"] - acr["corrected"]) * 100,
                "missing_sessions": acr["missing_cart_sessions"],
                "platform_detail": {k: (float(v) if isinstance(v, (int, float)) else v)
                                    for k, v in acr_platform.items()},
            },
            "gmv": {"observed": gmv["gmv_observed"], "corrected": gmv["gmv_corrected"],
                    "bias": gmv["relative_gap"]},
            "pv": {"observed": pv["pv_observed"], "dedup": pv["pv_deduplicated"],
                   "bias": -pv["inflation"]},
        },
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 68)
    print(f"日志 {len(df):,} 行｜{df['user_id'].nunique():,} 用户｜"
          f"{df['session_id'].nunique():,} 会话｜{df['event_time'].dt.date.nunique()} 天")
    print("-" * 68)
    for _, r in quality.iterrows():
        flag = "❌" if r["status"] == "fail" else "✅"
        print(f"{flag} {r['code']} {r['name']:<6} {r['detail']}")
    print("-" * 68)
    print("检出质量（对着注入真值）")
    print(detect[["defect", "granularity", "injected", "detected", "precision", "recall"]]
          .round(3).to_string(index=False))
    print("-" * 68)
    print("影响面：观测值 vs 校正值")
    for _, r in impact.iterrows():
        print(f"  {r['metric']:<10} {r['observed']:>10,.4f} → {r['corrected']:>10,.4f}"
              f"（{r['gap']}，{r['direction']}）")
    print("=" * 68)


if __name__ == "__main__":
    main()
