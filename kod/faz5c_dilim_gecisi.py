"""Faz 5c — Nufus agirlikli onluk dilim gecisi (K2 kanalinin yer degistirme etkisi).

Uc karsilastirma (revizyon, 29.09.2026):
  km -> duz enerji      : hiz + yardimci yuk etkisi (yukselti sifir)
  duz -> arazili enerji : yalnizca arazi etkisi
  km -> arazili enerji  : K2 kanalinin tamami (makaledeki ana sonuc)
Dilim tanimi Sekil 4 ile birebir aynidir (sekiller_uret.sekil4): kokenler (shapeID, il, ilce, nufus)
sirasinda, erisim degerine gore kararli siralama, birikimli nufus payinin onluk dilimi.
Spearman katsayisi koken duzeyi erisim DEGERLERI arasindadir (agirliksiz).
Girdi : ../veri/faz3_erisim_koken.csv, faz5_enerjik_koken.csv, faz5_enerjik_duz_koken.csv
Cikti : ../veri/faz5c_dilim_gecisi.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
a = ["shapeID", "il_ad", "ilce_ad", "nufus"]
KM = pd.read_csv(os.path.join(V, "faz3_erisim_koken.csv")).sort_values(a).reset_index(drop=True)
EN = pd.read_csv(os.path.join(V, "faz5_enerjik_koken.csv")).sort_values(a).reset_index(drop=True)
DZ = pd.read_csv(os.path.join(V, "faz5_enerjik_duz_koken.csv")).sort_values(a).reset_index(drop=True)
assert (KM.shapeID.values == EN.shapeID.values).all() and (KM.shapeID.values == DZ.shapeID.values).all()
P = KM.nufus.values


def dilim(x):
    o = np.argsort(x, kind="stable"); cw = np.cumsum(P[o]) / P.sum()
    d = np.empty(len(x), int); d[o] = np.minimum((cw * 10).astype(int), 9); return d


X = {"km": KM["BIRINCIL_NOM"].values, "duz": DZ["ENERJIK_NOM"].values, "arazili": EN["ENERJIK_NOM"].values}
D = {k: dilim(v) for k, v in X.items()}
sat = []
for s, h, ad in (("km", "duz", "hiz ve yardimci yuk"), ("duz", "arazili", "arazi"), ("km", "arazili", "K2 tamami")):
    fark = np.abs(D[s] - D[h])
    r = {"karsilastirma": "%s -> %s" % (s, h), "etki": ad,
         "dilim_degisen_%": 100 * P[fark > 0].sum() / P.sum(),
         "iki_ve_ustu_%": 100 * P[fark >= 2].sum() / P.sum(),
         "ort_kayma": float((P * fark).sum() / P.sum()),
         "spearman": float(spearmanr(X[s], X[h])[0])}
    sat.append(r)
    print("%-18s %-20s degisen %%%.1f | 2+ %%%.1f | ort kayma %.3f | Spearman %.3f"
          % (r["karsilastirma"], ad, r["dilim_degisen_%"], r["iki_ve_ustu_%"], r["ort_kayma"], r["spearman"]))
pd.DataFrame(sat).to_csv(os.path.join(V, "faz5c_dilim_gecisi.csv"), index=False, encoding="utf-8-sig")
print("Kaydedildi: faz5c_dilim_gecisi.csv")
