"""Faz 9b — Ilce regresyonu ve LISA icin uc deger saglamasi (revizyon, 29.09.2026).

Goreli degisim (A_kanal - A_nom) / A_nom, nominal erisimi cok kucuk ilcelerde patlar
(K2'de en buyuk deger 76.525). Bagimli degisken %1-%99 dilimlerinde kirpilarak faz9 ile ayni
model (nufus agirlikli EKK, HC3) ve ayni komsuluk (Queen, satir standart, 9.999 permutasyon)
yeniden kestirilir.
Girdi : ../veri/faz9_as2_ilce.csv, ../veri/faz0_ilce.gpkg
Cikti : ../veri/faz9b_kirpik.csv
"""
import os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
KAN = ["K1_kalite", "K2_topografya", "K3_tikaniklik", "K3_pi5", "K3_pi8", "K3_yaz", "K3_kis"]
D = pd.read_csv(os.path.join(V, "faz9_as2_ilce.csv")).set_index("shapeID")

G = gpd.read_file(os.path.join(V, "faz0_ilce.gpkg")).set_index("shapeID")
G = G.loc[G.index.intersection(D.index)]
w = Queen.from_dataframe(G, use_index=True)
tut = np.array([len(w.neighbors[k]) > 0 for k in G.index])
G2 = G[tut]
w2 = Queen.from_dataframe(G2, use_index=True); w2.transform = "r"

sat = []
for ad in KAN:
    y = D[ad + "_rel"]
    g = y.notna() & D.sege.notna() & (D.nufus > 0)
    lo, hi = y[g].quantile(0.01), y[g].quantile(0.99)
    yk = y.clip(lo, hi)
    X = sm.add_constant(pd.DataFrame({"sege": D.sege[g], "log_nufus": np.log(D.nufus[g])}))
    m = sm.WLS(yk[g], X, weights=D.nufus[g]).fit(cov_type="HC3")
    yl = yk.reindex(G2.index).fillna(0).values
    mi = Moran(yl, w2, permutations=9999)
    ml = Moran_Local(yl, w2, permutations=9999, seed=20260918)
    sig = ml.p_sim < 0.05
    sat.append({"kanal": ad, "ham_max": float(y[g].max()), "kirp_alt": lo, "kirp_ust": hi,
                "sege_kats": m.params["sege"], "sege_p": m.pvalues["sege"],
                "lognuf_kats": m.params["log_nufus"], "lognuf_p": m.pvalues["log_nufus"], "R2": m.rsquared,
                "moran_I": mi.I, "moran_p": mi.p_sim,
                "HH": int(((ml.q == 1) & sig).sum()), "LL": int(((ml.q == 3) & sig).sum()), "n": int(g.sum())})
    r = sat[-1]
    print("%-14s ham max %10.3f | SEGE %+.4f (p %.3f) lnP %+.4f (p %.3f) R2 %.3f | Moran %+.3f (p %.4f) HH %d LL %d"
          % (ad, r["ham_max"], r["sege_kats"], r["sege_p"], r["lognuf_kats"], r["lognuf_p"], r["R2"],
             r["moran_I"], r["moran_p"], r["HH"], r["LL"]))
pd.DataFrame(sat).to_csv(os.path.join(V, "faz9b_kirpik.csv"), index=False, encoding="utf-8-sig")
print("Kaydedildi: faz9b_kirpik.csv")
