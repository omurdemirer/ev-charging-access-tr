"""Faz 9c — Ilce duzeyi kanal etkisi: BOYUTSUZ endeks, artik otokorelasyonu, kumeli SH (revizyon, 30.09.2026).

4b: Eski bagimli degisken (A_kanal - A_nom)/A_nom, K1'de farkli birimleri (kisi basina oturum/sa ile kisi
basina soket) cikariyordu. Burada her olcu kendi nufus agirlikli ulusal ortalamasina bolunerek boyutsuz
endekse cevrilir:  r_d = (A_d^X / Abar^X) / (A_d^0 / Abar^0) - 1 .
Bu tanim olcek ve birimden bagimsizdir, Gini'nin olcek degismezligiyle tutarlidir; kanalin genel duzey
kaymasini degil DAGILIMSAL etkisini olcer. Nominal erisimi sifir olan ilceler tanimsizdir ve dislanir.
Bagimli degisken %1-%99 dilimlerinde kirpilir (faz9b ile ayni).
4c: (i) HC3 yaninda il duzeyinde KUMELI standart hatalar; (ii) regresyon ARTIKLARINDA Moran I;
(iii) LISA'da cok sayida sinama icin Benjamini-Hochberg duzeltmesi (q = 0,05) sonrasi kume sayilari.
Girdi : ../veri/faz3_erisim_ilce.csv, ../veri/faz5_enerjik_ilce.csv, ../veri/faz0_ilce.gpkg
Cikti : ../veri/faz9c_boyutsuz.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
from spreg import GM_Error_Het

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
KM = pd.read_csv(os.path.join(V, "faz3_erisim_ilce.csv")).set_index("shapeID")
EN = pd.read_csv(os.path.join(V, "faz5_enerjik_ilce.csv")).set_index("shapeID")
ortak = KM.index.intersection(EN.index); KM, EN = KM.loc[ortak], EN.loc[ortak]
P = KM.nufus


def endeks(x):
    return x / np.average(x, weights=P)


KAN = {"K1_kalite": (KM.BIRINCIL_K1, KM.BIRINCIL_NOM),
       "K2_enerji_mesafe": (EN.ENERJIK_NOM, KM.BIRINCIL_NOM),
       "K3_bugunku": (KM.BIRINCIL_NOM3, KM.BIRINCIL_NOM),
       "K3_5kat": (KM.BIRINCIL_NOM3p5, KM.BIRINCIL_NOM),
       "K3_8kat": (KM.BIRINCIL_NOM3p8, KM.BIRINCIL_NOM),
       "K3_yaz2025": (KM.BIRINCIL_NOM3Y25, KM.BIRINCIL_NOM),
       "K3_kis": (KM.BIRINCIL_NOMKIS if "BIRINCIL_NOMKIS" in KM else KM.BIRINCIL_NOM3KIS, KM.BIRINCIL_NOM)}

G = gpd.read_file(os.path.join(V, "faz0_ilce.gpkg")).set_index("shapeID")
G = G.loc[G.index.intersection(ortak)]


def ornek_agirlik(ix):
    """revizyon (02.10.2026): mekansal agirliklar YALNIZ gecerli gozlemlerden kurulur.
    Tanimsiz/eksik ilceler sifirla doldurulmaz; alt kumede komsusu kalmayan ilceler cikarilir."""
    ix = G.index.intersection(ix)
    for _ in range(5):
        ws = Queen.from_dataframe(G.loc[ix], use_index=True)
        kom = np.array([len(ws.neighbors[k]) > 0 for k in ix])
        if kom.all():
            break
        ix = ix[kom]
    ws.transform = "r"
    return ix, ws


def bh(p, q=0.05):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    esik = q * np.arange(1, m + 1) / m
    k = np.nonzero(p[o] <= esik)[0]
    kabul = np.zeros(m, bool)
    if len(k):
        kabul[o[:k.max() + 1]] = True
    return kabul


sat = []
for ad, (x, x0) in KAN.items():
    gec = (x0 > 0)
    r = (endeks(x) / endeks(x0) - 1).where(gec)
    g = r.notna() & KM.sege_skor.notna()
    lo, hi = r[g].quantile(0.01), r[g].quantile(0.99)
    y_tam = r[g].clip(lo, hi)
    # ortak ornek: gecerli endeks + gelismislik skoru + alt kumede en az bir komsu
    ix, ws = ornek_agirlik(y_tam.index)
    y = y_tam.loc[ix]
    X = sm.add_constant(pd.DataFrame({"sege": KM.sege_skor[ix], "log_nufus": np.log(P[ix])}))
    m_hc3 = sm.WLS(y, X, weights=P[ix]).fit(cov_type="HC3")
    m_kum = sm.WLS(y, X, weights=P[ix]).fit(cov_type="cluster", cov_kwds={"groups": KM.il_ad[ix].astype("category").cat.codes})
    m_ols = sm.OLS(y, X).fit(cov_type="HC3")                     # agirliksiz temel (SEM ile ayni ornek)
    mi_art = Moran(m_hc3.resid.values, ws, permutations=9999)
    mi = Moran(y.values, ws, permutations=9999)
    ml = Moran_Local(y.values, ws, permutations=9999, seed=20260918)
    anl = ml.p_sim < 0.05; anl_bh = bh(ml.p_sim)
    Xs = np.column_stack([KM.sege_skor[ix].values, np.log(P[ix].values)])
    se = GM_Error_Het(y.values.reshape(-1, 1), Xs, w=ws, name_x=["sege", "log_nufus"])
    zs = dict(zip(se.name_x, se.z_stat))
    r_ = {"kanal": ad, "n_gecerli": int(g.sum()), "n": len(ix),
          "sege_kats": m_hc3.params["sege"], "sege_p_hc3": m_hc3.pvalues["sege"], "sege_p_kumeli": m_kum.pvalues["sege"],
          "lognuf_kats": m_hc3.params["log_nufus"], "lognuf_p_hc3": m_hc3.pvalues["log_nufus"],
          "lognuf_p_kumeli": m_kum.pvalues["log_nufus"], "R2": m_hc3.rsquared,
          "ols_sege_kats": m_ols.params["sege"], "ols_sege_p": m_ols.pvalues["sege"],
          "sem_sege_kats": float(se.betas[1][0]), "sem_sege_p": float(zs["sege"][1]), "sem_lambda": float(se.betas[-1][0]),
          "moran_I": mi.I, "moran_p": mi.p_sim, "artik_moran_I": mi_art.I, "artik_moran_p": mi_art.p_sim,
          "HH": int(((ml.q == 1) & anl).sum()), "LL": int(((ml.q == 3) & anl).sum()),
          "HH_bh": int(((ml.q == 1) & anl_bh).sum()), "LL_bh": int(((ml.q == 3) & anl_bh).sum())}
    sat.append(r_)
    print("%-16s n=%d/%d | SEGE WLS %+.4f (p %.3f/%.3f) OLS %+.4f (p %.3f) SEM %+.4f (p %.3f, lam %.2f) | lnP %+.4f (p %.3f) "
          "| R2 %.3f | Moran %.3f (p %.4f) | artik %.3f (p %.4f) | HH/LL %d/%d, BH %d/%d"
          % (ad, r_["n"], r_["n_gecerli"], r_["sege_kats"], r_["sege_p_hc3"], r_["sege_p_kumeli"], r_["ols_sege_kats"],
             r_["ols_sege_p"], r_["sem_sege_kats"], r_["sem_sege_p"], r_["sem_lambda"], r_["lognuf_kats"],
             r_["lognuf_p_kumeli"], r_["R2"], r_["moran_I"], r_["moran_p"], r_["artik_moran_I"], r_["artik_moran_p"],
             r_["HH"], r_["LL"], r_["HH_bh"], r_["LL_bh"]), flush=True)
pd.DataFrame(sat).to_csv(os.path.join(V, "faz9c_boyutsuz.csv"), index=False, encoding="utf-8-sig")
print("Kaydedildi: faz9c_boyutsuz.csv")
