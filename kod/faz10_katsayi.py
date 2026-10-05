"""Faz 10 — AS5 icin ILCE x ISTASYON katsayi matrisi (optimizasyonun girdisi).

TEMEL OZDESLIK: 2SFCA'da erisim arz vektorunde DOGRUSALDIR (talep paydasi yalnizca havzaya ve
nufusa baglidir, arza DEGIL):
    A_o = SUM_j (S_j / talep_j) * W_oj
Ilce duzeyine nufus agirlikli toplanirsa:
    A_d = SUM_{o in d} (P_o/P_d) A_o = SUM_j S_j * kat[d,j],
    kat[d,j] = (1/talep_j) * SUM_{o in d} (P_o/P_d) * W_oj
Bu sayede optimizasyon, her aday cozum icin Dijkstra tekrarlamadan DOGRUSAL bir modele iner.
AC ve DC ayri havuzlar oldugu icin katsayi TUR BAZINDA tutulur.

Iki kume hesaplanir:
  (1) MEVCUT istasyonlar  -> soket EKLEME karari (merkezde K3'u gideren secenek)
  (2) ADAY konumlar       -> YENI istasyon karari (cevrede K1-K2'yi gideren secenek)
      Adaylar: tum ilce merkezleri (ilce_merkez.csv). Halihazirda istasyonu olan ilcelerde de
      aday tutulur; secim optimizasyona birakilir.

ONEMLI VARSAYIM: talep_j (2SFCA paydasi) arzdan bagimsizdir, ama YENI istasyon eklenince
mevcut istasyonlarin paydalari DEGISMEZ — bu 2SFCA'nin tanimi geregi boyledir (payda yalnizca
havzadaki nufustur). Rekabet/ikame etkisi modellenmez; bu, yontemin bilinen sinirlamasidir ve
makalede acikca yazilacaktir.

Girdi : ../veri/osm/*, ../veri/istasyon_duzeltilmis.csv, ../veri/osm/ilce_merkez.csv, ../veri/faz0_ilce.csv
Cikti : ../veri/faz10_kat_mevcut.npz, ../veri/faz10_kat_aday.npz, ../veri/faz10_indeks.json
Kullanim: python faz10_katsayi.py [--deneme 200]
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
HAVZA = {"AC": 5 * MIL, "DC": 25 * MIL}          # BIRINCIL havza (Tal+2020)
deneme = int(sys.argv[sys.argv.index("--deneme") + 1]) if "--deneme" in sys.argv else 0
t0 = time.time()


def W_gauss(d, p):
    return np.where(d <= p, (np.exp(-d * d / (2 * p * p)) - np.exp(-0.5)) / (1 - np.exp(-0.5)), 0.0)


K = pd.read_csv(os.path.join(OSM, "izgara_koken.csv"), index_col="o")
ag = Ag(OSM)
sira = np.argsort(K.dugum.values, kind="stable")
o_dugum = K.dugum.values[sira].astype(np.int64)
o_pop = K.nufus.values[sira]
o_bagkm = K.baglanma_km.values[sira]
o_sid = K.shapeID.values[sira]
ptr = np.searchsorted(o_dugum, np.arange(ag.N + 1))
Gk_r = ag.Gk.T.tocsr()

ilce = pd.read_csv(os.path.join(V, "faz0_ilce.csv"))
sid_list = list(pd.unique(o_sid))
sid_i = {s: i for i, s in enumerate(sid_list)}
o_di = np.array([sid_i[s] for s in o_sid], dtype=np.int64)
P_d = np.bincount(o_di, weights=o_pop, minlength=len(sid_list))
print("Koken %d | ilce %d | nufus %.0f | %.0f sn" % (len(K), len(sid_list), o_pop.sum(), time.time() - t0), flush=True)


def kat_hesapla(lon, lat, tipler, etiket):
    """tipler: her nokta icin islenecek tur listesi ({'AC','DC'} altkumesi)."""
    hs = ag.kenara_bagla(np.asarray(lon), np.asarray(lat))
    kk = pd.Series(list(zip(hs["ga"], hs["gb"])))
    gruplar = kk.groupby(kk).groups
    anah = list(gruplar)
    if deneme:
        rng = np.random.default_rng(20260918)
        anah = [anah[i] for i in rng.choice(len(anah), size=min(deneme, len(anah)), replace=False)]
    print("[%s] nokta %d | benzersiz kenar %d (islenecek %d)" % (etiket, len(lon), len(gruplar), len(anah)), flush=True)
    RD, RJ, RT, RV = [], [], [], []
    for gi, key in enumerate(anah):
        idx = np.asarray([s for s in gruplar[key] if tipler[s]])   # soketsiz istasyonlar atlanir
        if not len(idx):
            continue
        ga, gb = int(key[0]), int(key[1])
        lim = max(HAVZA[t] for s in idx for t in tipler[s]) + 0.1
        Dk = dijkstra(Gk_r, directed=True, indices=[ga, gb], limit=lim)
        ulas = np.nonzero(np.isfinite(Dk[0]) | np.isfinite(Dk[1]))[0]
        if not len(ulas):
            continue
        say = ptr[ulas + 1] - ptr[ulas]
        ulas = ulas[say > 0]; say = say[say > 0]
        if not len(ulas):
            continue
        oi = np.repeat(ptr[ulas], say) + (np.arange(say.sum()) - np.repeat(np.cumsum(say) - say, say))
        dn = np.repeat(np.arange(len(ulas)), say)
        for s in idx:
            fr = hs["frac"][s]
            ka = Dk[0][ulas] + fr * hs["e_km"][s] if hs["fwd"][s] else np.full(len(ulas), np.inf)
            kb = Dk[1][ulas] + (1 - fr) * hs["e_km"][s] if hs["rev"][s] else np.full(len(ulas), np.inf)
            d_km = np.minimum(ka, kb)[dn] + o_bagkm[oi] + hs["baglanma_km"][s]
            for t in tipler[s]:
                W = W_gauss(d_km, HAVZA[t])
                nz = W > 0
                if not nz.any():
                    continue
                ii, w = oi[nz], W[nz]
                talep = float(np.sum(o_pop[ii] * w))
                if talep <= 0:
                    continue
                # kat[d,j] = (1/talep) * SUM_{o in d} (P_o/P_d) W_oj
                pay = np.bincount(o_di[ii], weights=o_pop[ii] * w, minlength=len(sid_list))
                nzd = np.nonzero(pay)[0]
                v = pay[nzd] / np.maximum(P_d[nzd], 1e-9) / talep
                RD.append(nzd); RJ.append(np.full(len(nzd), s, dtype=np.int64))
                RT.append(np.full(len(nzd), 0 if t == "AC" else 1, dtype=np.int8)); RV.append(v)
        if gi % 500 == 0:
            print("  [%s] kenar %d/%d | %.0f sn" % (etiket, gi, len(anah), time.time() - t0), flush=True)
    return (np.concatenate(RD), np.concatenate(RJ), np.concatenate(RT), np.concatenate(RV))


# --- (1) mevcut istasyonlar ---
S = pd.read_csv(os.path.join(V, "istasyon_duzeltilmis.csv"))
S = S[(S.hizmet == "HALKA_ACIK") & ~S.ada.astype(bool)].reset_index(drop=True)
sk = pd.read_csv(os.path.join(V, "faz2c_soket_lambda.csv"), usecols=["ist_no", "tip", "P_etkin"])
g_ = sk.groupby(["ist_no", "tip"]).agg(n=("P_etkin", "size")).unstack(fill_value=0)
n_ac = g_[("n", "AC")].reindex(S.ist_no).fillna(0).values.astype(int)
n_dc = g_[("n", "DC")].reindex(S.ist_no).fillna(0).values.astype(int)
tip_m = [tuple(t for t, n in (("AC", n_ac[i]), ("DC", n_dc[i])) if n > 0) for i in range(len(S))]
d1, j1, t1, v1 = kat_hesapla(S.lon.values, S.lat.values, tip_m, "MEVCUT")
np.savez_compressed(os.path.join(V, "faz10_kat_mevcut.npz"), d=d1, j=j1, t=t1, v=v1,
                    n_ac=n_ac, n_dc=n_dc, P_d=P_d)
print("[MEVCUT] katsayi satiri %d | %.0f sn" % (len(v1), time.time() - t0), flush=True)

# --- (2) aday konumlar: ilce merkezleri (her ikisi de mumkun: AC ve DC) ---
M = pd.read_csv(os.path.join(OSM, "ilce_merkez.csv"))
M = M[~M.ada.astype(bool)].reset_index(drop=True)
tip_a = [("AC", "DC")] * len(M)
d2, j2, t2, v2 = kat_hesapla(M.lon.values, M.lat.values, tip_a, "ADAY")
np.savez_compressed(os.path.join(V, "faz10_kat_aday.npz"), d=d2, j=j2, t=t2, v=v2, P_d=P_d)
print("[ADAY] katsayi satiri %d | %.0f sn" % (len(v2), time.time() - t0), flush=True)

json.dump({"sid": sid_list, "ist_no": S.ist_no.tolist(),
           "aday_sid": M.shapeID.tolist(), "aday_il": M.il_ad.tolist(), "aday_ilce": M.ilce_ad.tolist(),
           "havza_km": HAVZA},
          open(os.path.join(V, "faz10_indeks.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("\nToplam sure %.0f sn" % (time.time() - t0))
