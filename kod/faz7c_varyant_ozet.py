"""Faz 7c — Enerji-mesafe kanalinin varyantlar arasinda ozeti (revizyon).

Her varyant icin (enerji dosyasi + karsilik gelen km havzasi oneki):
  gini_km, gini_en   : NOM yapilandirmada koken Gini'si (km havzasi, enerji havzasi)
  dGini              : gini_en - gini_km (dogrudan fark; ozdeki 0,021'in tanimi)
  K2_shapley         : enerji-mesafe kanalinin Shapley payi (bugunku yuk; faz7b ile ayni)
  dilim_%, dilim2_%  : km -> enerji gecisinde dilim degistiren nufus (faz5c/faz5d temel yontemi), 2+ dilim
  spearman_ilce      : ilce duzeyi enerji erisiminin TEMEL enerji erisimiyle Spearman sira korelasyonu
  alt10_ortak_%      : en dusuk erisimli %10 ilcenin temel listeyle ortakligi (siralamanin politika ucu)
Varyantlar dosyadan okunur; dosyasi henuz olmayanlar atlanir.
Cikti : ../veri/faz7c_varyant_ozet.csv
"""
import os, sys, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
anah = ["shapeID", "il_ad", "ilce_ad", "nufus"]
# (etiket, grup, km oneki, enerji dosyasi son eki)
VAR = [("temel", "temel", "BIRINCIL", ""),
       ("havza 3 mil (AC=DC)", "havza", "3mil", "_h3mil"),
       ("havza AC 15 / DC 25 mil", "havza", "AC15mil", "_hAC15mil"),
       ("butce: kenar ortalamasi", "butce", "BIRINCIL", "_bkenarort"),
       ("butce: -%10", "butce", "BIRINCIL", "_bm10"),
       ("butce: +%10", "butce", "BIRINCIL", "_bp10"),
       ("tunel/kopru enterpolasyonu", "arazi", "BIRINCIL", "_tunel"),
       ("kutle 1196 kg", "arac", "BIRINCIL", "_m1196"),
       ("kutle 1994 kg", "arac", "BIRINCIL", "_m1994"),
       ("hiz x0,9", "arac", "BIRINCIL", "_v0.9"),
       ("hiz x1,1", "arac", "BIRINCIL", "_v1.1"),
       ("aktarma verimi 0,782", "arac", "BIRINCIL", "_e0.782"),
       ("aktarma verimi 0,874", "arac", "BIRINCIL", "_e0.874")]

KMF = pd.read_csv(os.path.join(V, "faz3_erisim_koken.csv")).sort_values(anah).reset_index(drop=True)
P = KMF.nufus.values.astype(float); w = P / P.sum()


def gini(x):
    o = np.argsort(x, kind="stable"); cw = np.cumsum(w[o]); cx = np.cumsum((w * x)[o]) / (w * x).sum()
    return 1 - np.sum(np.diff(np.r_[0, cw]) * (cx + np.r_[0, cx[:-1]]))


def dilim(x):
    o = np.argsort(x, kind="stable"); cw = np.cumsum(P[o]) / P.sum(); d = np.empty(len(x), int)
    d[o] = np.minimum((cw * 10).astype(int), 9); return d


def ilce(x):
    g = pd.DataFrame({"sid": KMF.shapeID.values, "px": P * x, "p": P}).groupby("sid").sum()
    return g.px / g.p


def shapley_gini(km_on, en):
    v = {}
    for c in itertools.product([0, 1], repeat=3):
        k = ("K1" if c[0] else "NOM") + ("3" if c[2] else "")
        v[c] = gini(en["ENERJIK_" + k].values if c[1] else KMF[km_on + "_" + k].values)
    fakt = [1, 1, 2, 6]; t = 0.0
    for S in itertools.product([0, 1], repeat=3):
        if S[1]:
            continue
        Si = list(S); Si[1] = 1; k = sum(S)
        t += fakt[k] * fakt[2 - k] / 6 * (v[tuple(Si)] - v[S])
    return t


sat = []; taban = None
for et, grp, km_on, ek in VAR:
    f = os.path.join(V, "faz5_enerjik%s_koken.csv" % ek)
    if not os.path.exists(f):
        print("atlandi (dosya yok):", et); continue
    EN = pd.read_csv(f).sort_values(anah).reset_index(drop=True)
    assert (EN.shapeID.values == KMF.shapeID.values).all()
    x, y = KMF[km_on + "_NOM"].values, EN.ENERJIK_NOM.values
    dx, dy = dilim(x), dilim(y); fd = np.abs(dx - dy)
    ie = ilce(y)
    if taban is None:
        taban = ie
    a10 = set(taban.nsmallest(int(round(0.1 * len(taban)))).index); b10 = set(ie.nsmallest(len(a10)).index)
    sat.append({"varyant": et, "grup": grp, "gini_km": gini(x), "gini_en": gini(y), "dGini": gini(y) - gini(x),
                "K2_shapley": shapley_gini(km_on, EN),
                "dilim_%": 100 * P[fd > 0].sum() / P.sum(), "dilim2_%": 100 * P[fd >= 2].sum() / P.sum(),
                "spearman_ilce": spearmanr(taban.values, ie.reindex(taban.index).values).correlation,
                "alt10_ortak_%": 100 * len(a10 & b10) / len(a10)})
R = pd.DataFrame(sat)
print(R.round(4).to_string(index=False))
R.to_csv(os.path.join(V, "faz7c_varyant_ozet.csv"), index=False, encoding="utf-8-sig")
