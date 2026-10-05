"""Faz 7 — UC KANALIN SHAPLEY AYRISTIRMASI (AS3).

Kanallar (her biri ikili anahtar):
  K1 : arz tanimi   NOM (soket sayisi)        -> K1 (saatlik oturum kapasitesi)
  K2 : havza        KILOMETRE (DC 40,23/AC 8,05) -> ENERJI (nufus-eslesmeli kalibreli butce)
  K3 : tikaniklik   yok                        -> etkin arz x (1 - Erlang C)
2^3 = 8 yapilandirma. v(S) = ilgili yapilandirmanin Gini'si; toplam etki v(N) - v(bos).
Shapley:  phi_i = SUM_{S subset N\\{i}} |S|!(n-|S|-1)!/n! [v(S+i) - v(S)]
n=3 icin agirliklar: |S|=0 -> 1/3, |S|=1 -> 1/6, |S|=2 -> 1/3. SUM phi_i = v(N) - v(bos) (tam ayrisma).

NEDEN 8 KOSUM DEGIL 2: 2SFCA'da A arz vektorunde DOGRUSALDIR (talep paydasi yalnizca havzaya
baglidir). Bu yuzden sabit havzada tum arz tanimlari ayni Dijkstra gecisinde hesaplanir.
-> km havzasi icin 1 kosum, enerji havzasi icin 1 kosum yeter.

Girdi : ../veri/faz3_erisim_koken.csv (km havzasi), ../veri/faz5_enerjik_koken.csv (enerji havzasi)
        Her ikisinde de arz tanimlari: NOM, K1, NOM3, K13 (ve pi senaryolari).
Cikti : ../veri/faz7_shapley.csv
Kullanim: python faz7_shapley.py [--pi 1]      (K3 yuk senaryosu: 1, 5, 8)
"""
import sys, os, itertools
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
PI = sys.argv[sys.argv.index("--pi") + 1] if "--pi" in sys.argv else "1"
SON = "" if PI == "1" else "p%s" % PI
if "--k3" in sys.argv:                      # AS4 etiketi (Y25 / KIS / Y26)
    SON = sys.argv[sys.argv.index("--k3") + 1]
ENS = sys.argv[sys.argv.index("--enerji") + 1] if "--enerji" in sys.argv else ""


def gini_w(x, w):
    o = np.argsort(x); x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    if (x * w).sum() == 0:
        return np.nan
    cw = np.r_[0, np.cumsum(w) / w.sum()]; cx = np.r_[0, np.cumsum(x * w) / (x * w).sum()]
    return 1 - np.sum((cw[1:] - cw[:-1]) * (cx[1:] + cx[:-1]))


KM = pd.read_csv(os.path.join(V, "faz3_erisim_koken.csv"))
EN = pd.read_csv(os.path.join(V, "faz5_enerjik%s_koken.csv" % ENS))
anah = ["shapeID", "il_ad", "ilce_ad", "nufus"]
KM = KM.sort_values(anah).reset_index(drop=True)
EN = EN.sort_values(anah).reset_index(drop=True)
assert (KM.shapeID.values == EN.shapeID.values).all() and np.allclose(KM.nufus.values, EN.nufus.values), \
    "koken siralamalari ortusmuyor"
P = KM.nufus.values

# (K1, K2, K3) -> sutun adi
def sut(k1, k2, k3):
    arz = ("K1" if k1 else "NOM") + (("3" + SON) if k3 else "")
    return ("ENERJIK_" if k2 else "BIRINCIL_") + arz


eksik = [sut(*c) for c in itertools.product([0, 1], repeat=3)
         if sut(*c) not in (EN.columns if c[1] else KM.columns)]
if eksik:
    print("EKSIK SUTUNLAR (once faz3/faz5'i genisletilmis ARZ ile kosun):")
    for e in eksik:
        print("   ", e)
    print("\nMevcut km sutunlari    :", [c for c in KM.columns if c.startswith("BIRINCIL_")])
    print("Mevcut enerji sutunlari:", [c for c in EN.columns if c.startswith("ENERJIK_")])
    sys.exit(1)

v = {}
for c in itertools.product([0, 1], repeat=3):
    kay = EN if c[1] else KM
    v[c] = gini_w(kay[sut(*c)].values, P)

n = 3
fakt = [1, 1, 2, 6]
phi = {}
for i in range(n):
    t = 0.0
    for S in itertools.product([0, 1], repeat=n):
        if S[i] == 1:
            continue
        Si = list(S); Si[i] = 1; Si = tuple(Si)
        k = sum(S)
        w = fakt[k] * fakt[n - k - 1] / fakt[n]
        t += w * (v[Si] - v[S])
    phi[i] = t

ad = {0: "K1 (kalite/oturum)", 1: "K2 (enerji-mesafe)", 2: "K3 (tikaniklik, pi=%s)" % PI}
print("=== 8 YAPILANDIRMA (Gini, koken duzeyi, nufus agirlikli) ===")
print("K1 K2 K3 |   Gini")
for c in sorted(v, key=lambda x: (sum(x), x)):
    print(" %d  %d  %d | %.4f   [%s]" % (c[0], c[1], c[2], v[c], sut(*c)))
top = v[(1, 1, 1)] - v[(0, 0, 0)]
print("\nToplam etki v(1,1,1) - v(0,0,0) = %+.4f" % top)
mut = sum(abs(phi[k]) for k in range(n))
print("  NET pay: kanallar birbirini goturdugu icin %100'u asabilir/negatif olabilir.")
print("  MUTLAK pay = |phi_i| / SUM|phi| : kanalin toplam HAREKETTEKI agirligi.")
print("\n=== SHAPLEY PAYLARI ===")
for i in range(n):
    print("  %-26s %+.4f | net %%%7.1f | mutlak %%%5.1f"
          % (ad[i], phi[i], 100 * phi[i] / top if top else np.nan, 100 * abs(phi[i]) / mut))
print("  %-26s %+.4f" % ("TOPLAM (kontrol)", sum(phi.values())))
assert abs(sum(phi.values()) - top) < 1e-9, "Shapley tam ayrismiyor!"
print("  ✓ tam ayrisma dogrulandi (|fark| < 1e-9)")

pd.DataFrame([{"kanal": ad[i], "shapley": phi[i],
               "net_pay_%": 100 * phi[i] / top if top else np.nan,
               "mutlak_pay_%": 100 * abs(phi[i]) / mut} for i in range(n)] +
             [{"kanal": "TOPLAM", "shapley": top, "net_pay_%": 100.0, "mutlak_pay_%": 100.0}]).to_csv(
    os.path.join(V, "faz7_shapley%s%s.csv" % (SON, ENS)), index=False, encoding="utf-8-sig")
pd.DataFrame([{"K1": c[0], "K2": c[1], "K3": c[2], "sutun": sut(*c), "gini_koken": v[c]}
              for c in sorted(v, key=lambda x: (sum(x), x))]).to_csv(
    os.path.join(V, "faz7_yapilandirma%s%s.csv" % (SON, ENS)), index=False, encoding="utf-8-sig")
print("\nKaydedildi: faz7_shapley%s%s.csv ve faz7_yapilandirma%s%s.csv" % (SON, ENS, SON, ENS))
