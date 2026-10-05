#!/bin/bash
cd "$(dirname "$0")/.."
for p in "" p5 p8; do for r in 0 1 2 3 4 5 6 7 8 9; do python faz7_shapley.py --k3 "${p}_mc$r" > /dev/null 2>&1 || echo "HATA ${p}_mc$r"; done; done
python - <<'PY'
import pandas as pd, os
V="../veri"; sat=[]
for p in ["", "p5", "p8"]:
    for r in range(10):
        d=pd.read_csv(os.path.join(V,"faz7_shapley%s_mc%d.csv"%(p,r)))
        sat.append({"pi":p or "p1","r":r,**{k:v for k,v in zip(["K1","K2","K3","TOP"],d.shapley)},
                    **{k+"_mut":v for k,v in zip(["K1","K2","K3"],d["mutlak_pay_%"])}})
D=pd.DataFrame(sat); D.to_csv(os.path.join(V,"faz7_mc_ozet.csv"),index=False,encoding="utf-8-sig")
print(D.groupby("pi").agg(["min","max"]).round(4).T.to_string())
PY
