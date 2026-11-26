# -*- coding: utf-8 -*-
import json, pathlib, sys
try:
    import json, pathlib, pandas as pd
    ROOT=pathlib.Path(__file__).resolve().parents[1]; m={}
    d=pd.read_csv(ROOT/"data/real/retailrocket_events_sample.csv")
    cols={c.lower():c for c in d.columns}
    ev=cols.get("event","event"); vis=cols.get("visitorid")
    m["events"]=int(len(d)); m["visitors"]=int(d[vis].nunique()) if vis else None
    vc=d[ev].value_counts().to_dict(); m["by_event"]={k:int(v) for k,v in vc.items()}
    v=vc.get("view",0); atc=vc.get("addtocart",0); tr=vc.get("transaction",0)
    m["view_to_cart"]=round(atc/v,4) if v else None; m["cart_to_transaction"]=round(tr/atc,4) if atc else None
    print(json.dumps(m,ensure_ascii=False,indent=2))
    _root=pathlib.Path(__file__).resolve().parents[1]
    (_root/'outputs').mkdir(exist_ok=True)
    (_root/'reports').mkdir(exist_ok=True)
    (_root/'outputs/real_metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    NOTES='''# 真实公开数据集分析概览\n\n本仓库附带真实公开数据集（见 data/real/SOURCES.md），运行 `python src/real_data_analysis.py` 复现以下真实数据统计：\n\n```json\n'+json.dumps(m,ensure_ascii=False,indent=2)+'\n```\n\n> data/ 与 outputs/metrics.json 的主结果为脱敏/仿真案例，用于稳定演示方法；本文件使用公开真实数据集验证方法可运行。'''
    (_root/'reports/real-data-notes.md').write_text(NOTES,encoding='utf-8')
    print(json.dumps(m,ensure_ascii=False,indent=2))
except Exception as e:
    print('REAL_ANALYSIS_ERROR', type(e).__name__, e); sys.exit(1)
