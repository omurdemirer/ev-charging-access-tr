"""Faz 9 — AS2: kanal duzeltmelerinin ILCE duzeyindeki etkisi.
   (a) Kanal basina fark (etkin - nominal) -> SEGE uzerine regresyon
   (b) Farkin mekansal kumelenmesi -> Moran's I + LISA

Kanal farklari (ilce duzeyi, nufus agirlikli erisim):
  d_K1 = A(K1, km)     - A(NOM, km)         kalite
  d_K2 = A(NOM, enerji)- A(NOM, km)         topografya
  d_K3 = A(NOM3, km)   - A(NOM, km)         tikaniklik (pi=1; ayrica pi5/pi8 ve mevsim)
Fark BAGIL olarak da verilir: d/A(NOM,km) -> olcek etkisi temizlenir.

Regresyon: d_rel ~ SEGE + log(nufus) + kentsellik(il merkezi kuklasi)
  HC3 saglam standart hatalar (heteroskedastisite: ilce buyuklukleri cok farkli).
LISA: geoBoundaries ADM2 poligonlarindan Queen komsulugu; ada/komsusuz ilceler dislanir.

Girdi : ../veri/faz3_erisim_ilce.csv, ../veri/faz5_enerjik*_ilce.csv, ../veri/faz0_ilce.gpkg
Cikti : ../veri/faz9_as2_ilce.csv, ekrana regresyon + LISA ozeti
Kullanim: python faz9_as2_ilce.py [--enerji _paux1.2]
"""
import sys, os, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
ENS = sys.argv[sys.argv.index("--enerji") + 1] if "--enerji" in sys.argv else ""

KM = pd.read_csv(os.path.join(V, "faz3_erisim_ilce.csv"))
EN = pd.read_csv(os.path.join(V, "faz5_enerjik%s_ilce.csv" % ENS))
KM = KM.set_index("shapeID"); EN = EN.set_index("shapeID")
ortak = KM.index.intersection(EN.index)
KM, EN = KM.loc[ortak], EN.loc[ortak]
print("Ilce %d | enerji dosyasi sonegi: '%s'" % (len(KM), ENS or "(taban)"))

taban = KM["BIRINCIL_NOM"]
D = pd.DataFrame(index=KM.index)
D["il_ad"], D["ilce_ad"] = KM.il_ad, KM.ilce_ad
D["nufus"], D["sege"] = KM.nufus, KM.sege_skor
D["A_nom"] = taban
KAN = {"K1_kalite": KM["BIRINCIL_K1"] - taban,
       "K2_topografya": EN["ENERJIK_NOM"] - taban,
       "K3_tikaniklik": KM["BIRINCIL_NOM3"] - taban}
for s, ad in (("p5", "K3_pi5"), ("p8", "K3_pi8"), ("Y25", "K3_yaz"), ("KIS", "K3_kis"), ("Y26", "K3_yaz2026")):
    k = "BIRINCIL_NOM3" + s
    if k in KM.columns:
        KAN[ad] = KM[k] - taban
for ad, v in KAN.items():
    D[ad] = v
    D[ad + "_rel"] = np.where(taban > 0, v / taban.replace(0, np.nan), np.nan)

import statsmodels.api as sm
print("\n=== (a) KANAL FARKI ~ SEGE + log(nufus)  [HC3 saglam s.h., nufus agirlikli] ===")
print("%-14s %10s %10s %10s %10s %8s" % ("kanal", "SEGE kats.", "p", "logNuf", "p", "R2"))
sat = []
for ad in KAN:
    y = D[ad + "_rel"]
    g = y.notna() & D.sege.notna() & (D.nufus > 0)
    X = sm.add_constant(pd.DataFrame({"sege": D.sege[g], "log_nufus": np.log(D.nufus[g])}))
    m = sm.WLS(y[g], X, weights=D.nufus[g]).fit(cov_type="HC3")
    sat.append({"kanal": ad, "sege_kats": m.params["sege"], "sege_p": m.pvalues["sege"],
                "lognuf_kats": m.params["log_nufus"], "lognuf_p": m.pvalues["log_nufus"],
                "R2": m.rsquared, "n": int(g.sum())})
    print("%-14s %+10.4f %10.2g %+10.4f %10.2g %8.3f" % (ad, m.params["sege"], m.pvalues["sege"],
                                                         m.params["log_nufus"], m.pvalues["log_nufus"], m.rsquared))
REG = pd.DataFrame(sat)

# --- (b) LISA ---
print("\n=== (b) MEKANSAL KUMELENME (Queen komsulugu, 9999 permutasyon) ===")
try:
    import geopandas as gpd
    from libpysal.weights import Queen
    from esda.moran import Moran, Moran_Local
    G = gpd.read_file(os.path.join(V, "faz0_ilce.gpkg")).set_index("shapeID")
    G = G.loc[G.index.intersection(D.index)]
    Dg = D.loc[G.index]
    w = Queen.from_dataframe(G, use_index=True)
    yalniz = [i for i, k in enumerate(w.neighbors) if len(w.neighbors[k]) == 0]
    print("Poligon %d | komsusuz (ada) %d -> dislaniyor" % (len(G), len(yalniz)))
    tut = np.array([len(w.neighbors[k]) > 0 for k in G.index])
    G2, Dg2 = G[tut], Dg[tut]
    w2 = Queen.from_dataframe(G2, use_index=True); w2.transform = "r"
    print("%-14s %8s %8s | %6s %6s %6s %6s" % ("kanal", "Moran I", "p", "HH", "LL", "HL", "LH"))
    lis = []
    for ad in KAN:
        y = Dg2[ad + "_rel"].fillna(0).values
        mi = Moran(y, w2, permutations=9999)
        ml = Moran_Local(y, w2, permutations=9999, seed=20260918)
        sig = ml.p_sim < 0.05
        say = {q: int(((ml.q == q) & sig).sum()) for q in (1, 2, 3, 4)}   # 1 HH, 2 LH, 3 LL, 4 HL
        lis.append({"kanal": ad, "moran_I": mi.I, "moran_p": mi.p_sim,
                    "HH": say[1], "LL": say[3], "HL": say[4], "LH": say[2]})
        print("%-14s %+8.4f %8.4f | %6d %6d %6d %6d" % (ad, mi.I, mi.p_sim, say[1], say[3], say[4], say[2]))
        for q, et in ((1, "HH"), (3, "LL")):
            D.loc[Dg2.index[(ml.q == q) & sig], ad + "_lisa"] = et
    LIS = pd.DataFrame(lis)
except Exception as e:
    print("LISA atlandi:", type(e).__name__, e)
    LIS = pd.DataFrame()

D.to_csv(os.path.join(V, "faz9_as2_ilce%s.csv" % ENS), encoding="utf-8-sig")
REG.to_csv(os.path.join(V, "faz9_regresyon%s.csv" % ENS), index=False, encoding="utf-8-sig")
if len(LIS):
    LIS.to_csv(os.path.join(V, "faz9_lisa%s.csv" % ENS), index=False, encoding="utf-8-sig")
print("\nKaydedildi: faz9_as2_ilce%s.csv, faz9_regresyon%s.csv" % (ENS, ENS))
print("\nEn cok KAYBEDEN 8 ilce (K2 topografya, bagil):")
print(D.nsmallest(8, "K2_topografya_rel")[["il_ad", "ilce_ad", "nufus", "sege", "K2_topografya_rel"]].round(3).to_string(index=False))
