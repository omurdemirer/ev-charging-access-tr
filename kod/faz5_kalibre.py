"""Faz 5b — Enerji butcesinin NUFUS-ESLESMELI kalibrasyonu.

Sorun: E0 = 40,23 km * ebar seciminde ebar ULUSAL ORTALAMA tuketimdir (0,1579 kWh/km). Ama
tipik yol ortalamanin altinda tuketir (medyan kenar 0,1122 kWh/km); ortalama, yuksek tuketimli
kenarlarca sisirilir. Sonucta enerji havzasi km havzasindan sistematik olarak GENIS oluyor
(sifir erisim %2,15 -> %1,63). Seviye farki, dagilim karsilastirmasini kirletir.

Cozum: E0'i, AYNI NUFUSU kapsayacak sekilde sec.
  km havzasinda kapsanan nufus  W_km = sum_{o: km*(o) <= 40,23} P_o
  E0* :  sum_{o: E*(o) <= E0*} P_o  =  W_km
E*(o) = kokenden istasyona MINIMUM ENERJI (enerji-optimal yol), km*(o) = minimum mesafe.
Ornekleme: kalibrasyon sabiti bir ortalamadir; ORNEK istasyon kenariyla +-%1 dogrulukla bulunur.

Cikti: ekrana E0* (AC ve DC icin ayri) + esdeger ebar.
Kullanim: python faz5_kalibre.py [--ornek 800]
"""
import sys, os, time, json
import numpy as np, pandas as pd
from scipy.sparse.csgraph import dijkstra
from ag_ortak import Ag

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
OSM = os.path.join(V, "osm")
MIL = 1.609344
KM0 = {"AC": 5 * MIL, "DC": 25 * MIL}
EBAR = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri",
                                   "faz4_ebar%s.json" % os.environ.get("ENERJI_EK", "")), encoding="utf-8"))["ebar_kWh_per_km"]
MARJ = 5.0
ornek = int(sys.argv[sys.argv.index("--ornek") + 1]) if "--ornek" in sys.argv else 800
t0 = time.time()

K = pd.read_csv(os.path.join(OSM, "izgara_koken.csv"), index_col="o")
ag = Ag(OSM, enerji=True)
sira = np.argsort(K.dugum.values, kind="stable")
o_dugum = K.dugum.values[sira].astype(np.int64)
o_pop = K.nufus.values[sira]
o_bagkm = K.baglanma_km.values[sira]
ptr = np.searchsorted(o_dugum, np.arange(ag.N + 1))
Ge_r, Gk_r = ag.Ge.T.tocsr(), ag.Gk.T.tocsr()
phi = ag.phi
PHIMAX = float(phi.max())

S = pd.read_csv(os.path.join(V, "istasyon_duzeltilmis.csv"))
S = S[(S.hizmet == "HALKA_ACIK") & ~S.ada.astype(bool)].reset_index(drop=True)
hs = ag.kenara_bagla(S.lon.values, S.lat.values)
sk = pd.read_csv(os.path.join(V, "faz2c_soket_lambda.csv"), usecols=["ist_no", "tip", "P_etkin"])
g_ = sk.groupby(["ist_no", "tip"]).agg(n=("P_etkin", "size")).unstack(fill_value=0)
n_dc = g_[("n", "DC")].reindex(S.ist_no).fillna(0).values

kk = pd.Series(list(zip(hs["ga"], hs["gb"])))
gruplar = kk.groupby(kk).groups
anahtarlar = list(gruplar)
TOHUM = int(sys.argv[sys.argv.index("--tohum") + 1]) if "--tohum" in sys.argv else 20260918   # revizyon (02.10.2026): ornek oynakligi
rng = np.random.default_rng(TOHUM)
sec = rng.choice(len(anahtarlar), size=min(ornek, len(anahtarlar)), replace=False)
anahtarlar = [anahtarlar[i] for i in sec]
print("Ornek kenar %d / %d | %.0f sn" % (len(anahtarlar), len(gruplar), time.time() - t0), flush=True)

havuz = {"AC": [], "DC": []}
for gi, key in enumerate(anahtarlar):
    idx = np.asarray(gruplar[key])
    ga, gb = int(key[0]), int(key[1])
    tip = "DC" if (n_dc[idx] > 0).any() else "AC"
    ebar0 = EBAR
    Dk = dijkstra(Gk_r, directed=True, indices=[ga, gb], limit=KM0[tip] + 0.1)
    # kesin sinir (revizyon): butce 1,5*KM0*ebar0'a kadar aranir + potansiyel farki PHIMAX - phi(g)
    De = dijkstra(Ge_r, directed=True, indices=[ga, gb], limit=1.5 * KM0[tip] * ebar0 + PHIMAX - min(phi[ga], phi[gb]))
    ulas = np.nonzero(np.isfinite(Dk[0]) | np.isfinite(Dk[1]) | np.isfinite(De[0]) | np.isfinite(De[1]))[0]
    if not len(ulas):
        continue
    say = ptr[ulas + 1] - ptr[ulas]
    ulas = ulas[say > 0]; say = say[say > 0]
    if not len(ulas):
        continue
    oi = np.repeat(ptr[ulas], say) + (np.arange(say.sum()) - np.repeat(np.cumsum(say) - say, say))
    dn = np.repeat(np.arange(len(ulas)), say)
    s = int(idx[0])
    fr = hs["frac"][s]
    ka = Dk[0][ulas] + fr * hs["e_km"][s] if hs["fwd"][s] else np.full(len(ulas), np.inf)
    kb = Dk[1][ulas] + (1 - fr) * hs["e_km"][s] if hs["rev"][s] else np.full(len(ulas), np.inf)
    ea = (De[0][ulas] - phi[ulas] + phi[ga]) + fr * hs["e_kwh"][s] if hs["fwd"][s] else np.full(len(ulas), np.inf)
    eb = (De[1][ulas] - phi[ulas] + phi[gb]) + (1 - fr) * hs["e_kwh_r"][s] if hs["rev"][s] else np.full(len(ulas), np.inf)
    d_km = np.minimum(ka, kb)[dn] + o_bagkm[oi] + hs["baglanma_km"][s]
    d_kwh = np.maximum(np.minimum(ea, eb)[dn] + (o_bagkm[oi] + hs["baglanma_km"][s]) * ebar0, 0.0)
    ok = np.isfinite(d_km) | np.isfinite(d_kwh)
    havuz[tip].append(np.c_[o_pop[oi][ok], d_km[ok], d_kwh[ok]])
    if gi % 200 == 0:
        print("  %d/%d | %.0f sn" % (gi, len(anahtarlar), time.time() - t0), flush=True)

print("\n--- NUFUS-ESLESMELI KALIBRASYON ---")
sonuc = {}
for tip in ("AC", "DC"):
    if not havuz[tip]:
        continue
    M = np.vstack(havuz[tip])
    P, dkm, dkwh = M[:, 0], M[:, 1], M[:, 2]
    W_km = P[dkm <= KM0[tip]].sum()
    o = np.argsort(dkwh); cw = np.cumsum(P[o])
    j = int(np.searchsorted(cw, W_km))
    E0s = float(dkwh[o][min(j, len(o) - 1)])
    sonuc[tip] = E0s
    print("%s: km havzasi %.2f km -> kapsanan agirlik %.4g | ESLESEN E0* = %.4f kWh "
          "(esdeger ebar = %.4f kWh/km; kenar-ortalamasi ebar -> E0 = %.4f)"
          % (tip, KM0[tip], W_km, E0s, E0s / KM0[tip], KM0[tip] * EBAR))
    ic = dkwh <= E0s
    # tanilama: yalnizca HER IKI olcude de sonlu ve dkm>0 olan ciftler (inf/0'a bolme disarida)
    g = (dkm <= KM0[tip]) & np.isfinite(dkwh) & np.isfinite(dkm) & (dkm > 1e-6)
    print("   kapsanan cift orani: km %.3f vs enerji %.3f | ortalama gerceklesen oran %.4f kWh/km (n=%d)"
          % ((dkm <= KM0[tip]).mean(), ic.mean(), np.average(dkwh[g] / dkm[g], weights=P[g]), g.sum()))
json.dump({"E0_kalibre": sonuc, "yontem": "nufus-eslesmeli", "ornek_kenar": len(anahtarlar)},
          open(os.path.join(V, "faz5_kalibrasyon%s%s.json" % (os.environ.get("ENERJI_EK", ""), ("_tohum%d" % TOHUM) if "--tohum" in sys.argv else "")), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nToplam sure %.0f sn" % (time.time() - t0))
