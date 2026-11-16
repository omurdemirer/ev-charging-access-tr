"""Faz 5d — Dilim gecisinde esit degerli kokenlerin isleme bicimine duyarlilik (revizyon).

Temel yontem (faz5c): kokenler sabit bir siraya gore kararli siralanir, esit erisimli kokenler bu
siraya gore ardisik dilimlere dagilir. Burada uc alternatif sinanir:
  rastgele : esitler 20 farkli tohumla rastgele sirada (dagilimin ortalamasi ve araligi)
  grup     : esit degerli kokenlerin tamami, grubun birikimli nufus orta noktasinin dilimine atanir
  sifirsiz : iki havzanin herhangi birinde sifir erisimli kokenler disarida birakilir (esitlerin cogu sifirdir)
Girdi : ../veri/faz3_erisim_koken.csv, ../veri/faz5_enerjik_koken.csv
Cikti : ../veri/faz5d_esitlik_duyarliligi.csv
"""
import os, sys
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
a = ["shapeID", "il_ad", "ilce_ad", "nufus"]
KM = pd.read_csv(os.path.join(V, "faz3_erisim_koken.csv"), usecols=a + ["BIRINCIL_NOM"]).sort_values(a).reset_index(drop=True)
EN = pd.read_csv(os.path.join(V, "faz5_enerjik_koken.csv"), usecols=a + ["ENERJIK_NOM"]).sort_values(a).reset_index(drop=True)
assert (KM.shapeID.values == EN.shapeID.values).all()
P0 = KM.nufus.values
X0, Y0 = KM.BIRINCIL_NOM.values, EN.ENERJIK_NOM.values


def dilim_sira(x, P, o):
    cw = np.cumsum(P[o]) / P.sum(); d = np.empty(len(x), int)
    d[o] = np.minimum((cw * 10).astype(int), 9); return d


def dilim_grup(x, P):
    o = np.argsort(x, kind="stable"); xs = x[o]; cw = np.cumsum(P[o]) / P.sum()
    _, bas, say = np.unique(xs, return_index=True, return_counts=True)
    orta = np.empty(len(x))
    for b, c in zip(bas, say):
        lo = cw[b - 1] if b > 0 else 0.0
        orta[b:b + c] = 0.5 * (lo + cw[b + c - 1])
    d = np.empty(len(x), int); d[o] = np.minimum((orta * 10).astype(int), 9); return d


def oran(dx, dy, P):
    f = np.abs(dx - dy)
    return 100 * P[f > 0].sum() / P.sum(), 100 * P[f >= 2].sum() / P.sum()


def esit_pay(x, P):
    _, inv, say = np.unique(x, return_inverse=True, return_counts=True)
    return 100 * P[say[inv] > 1].sum() / P.sum()


sat = []
print("Esit degerli gruplardaki nufus payi: km %.2f%% (sifir %.2f%%), enerji %.2f%% (sifir %.2f%%)"
      % (esit_pay(X0, P0), 100 * P0[X0 == 0].sum() / P0.sum(), esit_pay(Y0, P0), 100 * P0[Y0 == 0].sum() / P0.sum()))
o = np.argsort(X0, kind="stable"), np.argsort(Y0, kind="stable")
t = oran(dilim_sira(X0, P0, o[0]), dilim_sira(Y0, P0, o[1]), P0)
sat.append({"yontem": "temel (kararli sira)", "degisen_%": t[0], "iki_ve_ustu_%": t[1]})
r = []
for s in range(20):
    g = np.random.default_rng(20261116 + s).permutation(len(X0))
    ox = g[np.argsort(X0[g], kind="stable")]; oy = g[np.argsort(Y0[g], kind="stable")]
    r.append(oran(dilim_sira(X0, P0, ox), dilim_sira(Y0, P0, oy), P0))
r = np.array(r)
sat.append({"yontem": "rastgele (20 tohum, ortalama)", "degisen_%": r[:, 0].mean(), "iki_ve_ustu_%": r[:, 1].mean(),
            "aralik": "%.1f-%.1f / %.1f-%.1f" % (r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max())})
t = oran(dilim_grup(X0, P0), dilim_grup(Y0, P0), P0)
sat.append({"yontem": "grup orta noktasi", "degisen_%": t[0], "iki_ve_ustu_%": t[1]})
m = (X0 > 0) & (Y0 > 0)
X1, Y1, P1 = X0[m], Y0[m], P0[m]
t = oran(dilim_sira(X1, P1, np.argsort(X1, kind="stable")), dilim_sira(Y1, P1, np.argsort(Y1, kind="stable")), P1)
sat.append({"yontem": "sifir erisimliler disarida (%.2f%% nufus)" % (100 - 100 * P1.sum() / P0.sum()),
            "degisen_%": t[0], "iki_ve_ustu_%": t[1]})
R = pd.DataFrame(sat)
print(R.to_string(index=False))
R.to_csv(os.path.join(V, "faz5d_esitlik_duyarliligi.csv"), index=False, encoding="utf-8-sig")
