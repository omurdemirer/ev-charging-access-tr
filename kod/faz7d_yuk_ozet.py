"""Faz 7d — tikaniklik kanalinin yuk duzeyi, 10 dk gostergesi ve talep dagitimina gore ozeti (revizyon).

Her K3 etiketi icin: nufus agirlikli ortalama erisim kaybi (nominal km yapilandirmasina gore) ve
faz7b_gostergeler<etiket>.csv'den K3'un Gini payi, mutlak payi, Atkinson mutlak payi ve alt %20'ye etkisi.
Girdi : ../veri/faz3_erisim_koken.csv, ../veri/faz7b_gostergeler<etiket>.csv (once faz7b calistirilir)
Cikti : ../veri/faz7d_yuk_gosterge_dagitim.csv
"""
import os, sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
ET = ["", "p1.5", "p3", "p5", "p8", "p12", "w", "wp1.5", "wp3", "wp5", "wp8", "wp12",
      "dsoket", "p5dsoket", "p8dsoket", "dilce_nufus", "p5dilce_nufus", "p8dilce_nufus"]
km = pd.read_csv(os.path.join(V, "faz3_erisim_koken.csv"), usecols=["nufus", "BIRINCIL_NOM"] + ["BIRINCIL_NOM3" + k for k in ET])
P = km.nufus.values
m = lambda c: (P * km[c]).sum() / P.sum()
sat = []
for k in ET:
    g = pd.read_csv(os.path.join(V, "faz7b_gostergeler%s.csv" % k)).set_index("gosterge")
    sat.append({"k3": k or "1", "ort_kayip_%": 100 * (1 - m("BIRINCIL_NOM3" + k) / m("BIRINCIL_NOM")),
                "gini_K3": g.loc["gini", "K3"], "gini_K3_%": g.loc["gini", "K3_mut_%"], "gini_K2": g.loc["gini", "K2"],
                "atk_K3_%": g.loc["atkinson", "K3_mut_%"], "alt20_K3": g.loc["alt20", "K3"]})
R = pd.DataFrame(sat)
print(R.round(4).to_string(index=False))
R.to_csv(os.path.join(V, "faz7d_yuk_gosterge_dagitim.csv"), index=False, encoding="utf-8-sig")
