"""81 ilin sarj tuketim payini, EPDK'nin yayimladigi ilk-10 il panelinden tahmin etme denemesi.

Girdi : ../veri/faz0_ilce.csv (ilce: nufus 2025, SEGE, halka acik istasyon/soket/DC/kW)
        ../veri/epdk_panel_il_top10.csv (2025-07 -> 2026-07, her ay ilk 10 il payi)
Model : log(tuketim payi) = a + b' X   (X: log nufus payi, log kW payi, log DC payi, SEGE)
Testler:
  1) Il-disarida capraz dogrulama (LOPO): her il modelden cikarilip tahmin edilir.
  2) KALAN TESTI (orneklem disi): her ay ilk-10 disindaki illerin tahmini toplam payi,
     gercek kalanla (100 - ilk-10 toplami) karsilastirilir. Model bu illeri hic gormez.
Uyari: orneklem kesik (yalnizca ilk-10 giriyor) -> kucuk illere tasima (ekstrapolasyon) riskli;
       kalan testi bu yanliligin buyuklugunu olcer.
"""
import sys, os
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")

g = pd.read_csv(os.path.join(V, "faz0_ilce.csv"))
g["w"] = g.sege_skor * g.nufus
il = g.groupby("il_ad").agg(nufus=("nufus", "sum"), kw=("kw", "sum"), dc=("dc", "sum"), w=("w", "sum"))
il["sege"] = il.w / il.nufus
for c, k in (("nufus", "pop"), ("kw", "kwp"), ("dc", "dcp")):
    il[k] = 100 * il[c] / il[c].sum()
    il["l" + k] = np.log(il[k].clip(lower=1e-3))

t = pd.read_csv(os.path.join(V, "epdk_panel_il_top10.csv"))
d = t.merge(il, left_on="il", right_index=True, how="left", validate="m:1")
d["ly"] = np.log(d.pay)
kalan = 100 - t.groupby("ay").pay.sum()                   # her ay ilk-10 disi gercek pay

MODELLER = {"nufus": ["lpop"], "nufus+kW": ["lpop", "lkwp"], "nufus+SEGE": ["lpop", "sege"],
            "nufus+DC": ["lpop", "ldcp"], "kW": ["lkwp"], "DC": ["ldcp"]}


def fit(df, X):
    A = np.c_[np.ones(len(df)), df[X].values]
    b, *_ = np.linalg.lstsq(A, df.ly.values, rcond=None)
    return b


def pred(df, X, b):
    return np.exp(np.c_[np.ones(len(df)), df[X].values] @ b)


sonuc, tahminler = [], {}
for ad, X in MODELLER.items():
    b = fit(d, X)
    r = d.ly - np.log(pred(d, X, b))
    r2 = 1 - r.var() / d.ly.var()
    # 1) LOPO
    e = []
    for p in d.il.unique():
        tr, te = d[d.il != p], d[d.il == p]
        e += list(te.ly - np.log(pred(te, X, fit(tr, X))))
    lopo = np.sqrt(np.mean(np.square(e)))
    # 2) kalan testi
    P = pd.Series(pred(il, X, b), index=il.index)
    tahminler[ad] = P
    oran = []
    for ay, grp in t.groupby("ay"):
        disari = P.drop(grp.il.values).sum()
        oran.append(disari / kalan[ay])
    sonuc.append(dict(model=ad, katsayilar=np.round(b, 3).tolist(), R2=round(r2, 2),
                      LOPO_log_hata=round(lopo, 2), LOPO_carpan=round(np.exp(lopo), 2),
                      kalan_tahmin_gercek_ort=round(np.mean(oran), 2),
                      kalan_min=round(min(oran), 2), kalan_max=round(max(oran), 2),
                      tum81_toplam=round(P.sum(), 1)))

R = pd.DataFrame(sonuc)
print("Gozlem: %d (%d il, %d ay) | gercek kalan pay: ort %.1f%%, aralik %.1f-%.1f%%\n"
      % (len(d), d.il.nunique(), t.ay.nunique(), kalan.mean(), kalan.min(), kalan.max()))
print(R.to_string(index=False))
print("\nOkuma: LOPO_carpan = bir ilin tahmininin tipik olarak kac kat sapabilecegi (1,00 = kusursuz).")
print("       kalan_tahmin_gercek = 1,00 ise model gorulmemis 71 ilin toplamini dogru buluyor.")

en = R.sort_values("LOPO_log_hata").iloc[0].model
P = tahminler[en] / tahminler[en].sum() * 100
out = il[["nufus", "pop", "kwp", "dcp", "sege"]].assign(tahmin_pay=P).sort_values("tahmin_pay", ascending=False)
out["tahmin_pay_bolu_kwp"] = out.tahmin_pay / out.kwp
out.round(3).to_csv(os.path.join(V, "il_tuketim_tahmin.csv"), encoding="utf-8-sig")
print("\nEn iyi LOPO modeli: %s -> 81 il tahmini: veri/il_tuketim_tahmin.csv (toplam 100'e olceklendi)" % en)
print(out.head(15).round(2).to_string())
