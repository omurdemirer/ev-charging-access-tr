"""Faz 5 — K2 (TOPOGRAFYA) kanali: ENERJI-MESAFE ile E2SFCA.

Fikir: Faz 3'te havza KILOMETRE ile tanimliydi (DC 40,23 km / AC 8,05 km). Burada ayni havza
ENERJI BUTCESI olarak yeniden tanimlanir: ulusal ortalama tuketim ebar = 0,1579 kWh/km ile
  E0_DC = 40,23 km * ebar = 6,353 kWh      E0_AC = 8,05 km * ebar = 1,271 kWh
Duz arazide ikisi AYNI havzayi verir. Engebeli arazide ayni butce daha kisa mesafeye yeter,
dolayisiyla erisim duser. Iki kosumun farki K2 kanalinin SAF etkisidir (arz tanimi sabit tutulur).

Yon asimetrisi: kWh yone baglidir (olculen ortalama |E(u,v)-E(v,u)| = 0,2598 kWh, ortalama kenar
enerjisinin iki kati). Ters graf uzerinde istasyondan geriye arama dogru yonu (koken -> istasyon) verir.

Potansiyel donusumu: Ge agirliklari w = E + phi(u) - phi(v) >= 0. Ters graf uzerinde istasyon g'den
yapilan aramada D(o) = E(o->g) + phi(o) - phi(g)  =>  E(o->g) = D(o) - phi(o) + phi(g).

Dijkstra limiti: D <= E0 + (phi(o) - phi(g)) oldugundan limit = E0 + MARJ alinir. MARJ = 5,0 kWh,
inis katsayisiyla ~1740 m rakim avantajina karsilik gelir; 40 km yaricapta bu buyuklukte fark
Turkiye'de pratikte yoktur.

Girdi : ../veri/osm/*, ../veri/istasyon_duzeltilmis.csv, ../veri/faz2c_*, ../veri/faz0_ilce.csv
Cikti : ../veri/faz5_enerji_koken.csv, faz5_enerji_ilce.csv, faz5_enerji_ozet.csv
Kullanim: python faz5_enerji_erisim.py [--deneme 150]
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
EBAR = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri",
                                   "faz4_ebar%s.json" % os.environ.get("ENERJI_EK", "")), encoding="utf-8"))["ebar_kWh_per_km"]
E0 = {"AC": 5 * MIL * EBAR, "DC": 25 * MIL * EBAR}
MARJ = 5.0
# --kalibre: nufus-eslesmeli butce (faz5_kalibre.py). Enerji havzasinin km havzasiyla AYNI
# nufusu kapsamasini saglar; boylece Gini farki SAF dagilim etkisidir (seviye kaymasi yok).
ETIKET = "ENERJI"
if "--kalibre" in sys.argv:
    # revizyon: ENERJI_BUTCE_EK verilirse butce BASKA bir senaryonun kalibrasyonundan alinir
    # (ornek: kis grafigi + yaz butcesi = SABIT batarya enerjisi). Cikti eki "_sabit<ek>" olur.
    _bek = os.environ.get("ENERJI_BUTCE_EK", os.environ.get("ENERJI_EK", ""))
    _k = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri",
                                     "faz5_kalibrasyon%s.json" % _bek), encoding="utf-8"))["E0_kalibre"]
    E0 = {t: float(_k[t]) for t in ("AC", "DC")}
    ETIKET = "ENERJIK"; EKD = os.environ.get("ENERJI_EK", "")
    if "ENERJI_BUTCE_EK" in os.environ:
        EKD += "_sabit" + (_bek or "birincil")
deneme = int(sys.argv[sys.argv.index("--deneme") + 1]) if "--deneme" in sys.argv else 0
t0 = time.time()


def W_gauss(d, p):
    return np.where(d <= p, (np.exp(-d * d / (2 * p * p)) - np.exp(-0.5)) / (1 - np.exp(-0.5)), 0.0)


def gini_w(x, w):
    o = np.argsort(x); x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    if (x * w).sum() == 0:
        return np.nan
    cw = np.r_[0, np.cumsum(w) / w.sum()]; cx = np.r_[0, np.cumsum(x * w) / (x * w).sum()]
    return 1 - np.sum((cw[1:] - cw[:-1]) * (cx[1:] + cx[:-1]))


def theil_w(x, w, grup):
    x = np.maximum(np.asarray(x, float), 1e-12); w = np.asarray(w, float) / np.sum(w)
    mu = np.sum(w * x); T = np.sum(w * x / mu * np.log(x / mu))
    df = pd.DataFrame({"x": x, "w": w, "g": grup})
    gk = df.groupby("g").apply(lambda q: pd.Series({"w": q.w.sum(), "mu": np.sum(q.w * q.x) / q.w.sum()}), include_groups=False)
    return T, np.sum(gk.w * gk.mu / mu * np.log(gk.mu / mu))


K = pd.read_csv(os.path.join(OSM, "izgara_koken.csv"), index_col="o")
ag = Ag(OSM, enerji=True)
sira = np.argsort(K.dugum.values, kind="stable")
o_dugum = K.dugum.values[sira].astype(np.int64)
o_pop = K.nufus.values[sira]
o_bagkwh = K.baglanma_km.values[sira] * EBAR       # baglanma bacagi duz kabul edilir
ptr = np.searchsorted(o_dugum, np.arange(ag.N + 1))
Ge_r = ag.Ge.T.tocsr()
phi = ag.phi
PHIMAX = float(phi.max())
print("Koken %d | dugum %d | nufus %.0f | E0: AC %.3f, DC %.3f kWh | %.0f sn"
      % (len(K), ag.N, o_pop.sum(), E0["AC"], E0["DC"], time.time() - t0), flush=True)

S = pd.read_csv(os.path.join(V, "istasyon_duzeltilmis.csv"))
S = S[(S.hizmet == "HALKA_ACIK") & ~S.ada.astype(bool)].reset_index(drop=True)
hs = ag.kenara_bagla(S.lon.values, S.lat.values)
par = json.load(open(os.path.join(V, "faz2c_lambda_param.json"), encoding="utf-8"))
sk = pd.read_csv(os.path.join(V, "faz2c_soket_lambda.csv"), usecols=["ist_no", "tip", "P_etkin"])
g_ = sk.groupby(["ist_no", "tip"]).agg(n=("P_etkin", "size"), kw=("P_etkin", "sum")).unstack(fill_value=0)
n_ac = g_[("n", "AC")].reindex(S.ist_no).fillna(0).values
n_dc = g_[("n", "DC")].reindex(S.ist_no).fillna(0).values
kw_ac = g_[("kw", "AC")].reindex(S.ist_no).fillna(0).values
kw_dc = g_[("kw", "DC")].reindex(S.ist_no).fillna(0).values
ARZ = {"NOM": {"AC": n_ac, "DC": n_dc},
       "K1": {"AC": n_ac * 60 / par["dk_oturum"]["AC"], "DC": n_dc * 60 / par["dk_oturum"]["DC"]},
       "K1g": {"AC": kw_ac, "DC": kw_dc}}
print("Istasyon %d | AC soketli %d, DC soketli %d" % (len(S), (n_ac > 0).sum(), (n_dc > 0).sum()), flush=True)

# --- K3 (tikaniklik) carpanlari: faz6 Erlang-C'den aninda hizmet olasiligi (1 - C) ---
def _k3_carpan(ist_no_dizi, pi_son="", kolon="aninda_hizmet"):
    yol = os.path.join(V, "faz6_istasyon_k3%s.csv" % pi_son)
    q = pd.read_csv(yol).set_index(["ist_no", "tip"])[kolon]
    return {t: q.reindex(pd.MultiIndex.from_product([ist_no_dizi, [t]])).fillna(1.0).values
            for t in ("AC", "DC")}


for _pi_son, _etiket in (("", ""), ("_pi5", "p5"), ("_pi8", "p8"),
                         ("_YAZ2025", "Y25"), ("_KIS", "KIS"), ("_YAZ2026", "Y26")):
    if not os.path.exists(os.path.join(V, "faz6_istasyon_k3%s.csv" % _pi_son)):
        continue
    _f = _k3_carpan(S.ist_no.values, _pi_son)
    ARZ["NOM3" + _etiket] = {t: ARZ["NOM"][t] * _f[t] for t in ("AC", "DC")}
    ARZ["K13" + _etiket] = {t: ARZ["K1"][t] * _f[t] for t in ("AC", "DC")}
# revizyon: (30.09.2026): K3_EK=1 ise 10 dk icinde hizmet gostergesi (K13w) ve Monte Carlo
# cekilisleri (K13_mc*) eklenir; yalniz birincil koşuda kullanilir (hesap yuku)
if os.environ.get("K3_EK") == "1":
    for _pi_son, _etiket in (("", ""), ("_pi5", "p5"), ("_pi8", "p8")):
        _q = pd.read_csv(os.path.join(V, "faz6_istasyon_k3%s.csv" % _pi_son), nrows=1).columns
        _f = _k3_carpan(S.ist_no.values, _pi_son, "hizmet10")
        ARZ["NOM3w" + _etiket] = {t: ARZ["NOM"][t] * _f[t] for t in ("AC", "DC")}
        ARZ["K13w" + _etiket] = {t: ARZ["K1"][t] * _f[t] for t in ("AC", "DC")}
        for _k in [c for c in _q if c.startswith("aninda_mc")]:
            _f = _k3_carpan(S.ist_no.values, _pi_son, _k)
            ARZ["NOM3" + _etiket + "_" + _k[7:]] = {t: ARZ["NOM"][t] * _f[t] for t in ("AC", "DC")}
            ARZ["K13" + _etiket + "_" + _k[7:]] = {t: ARZ["K1"][t] * _f[t] for t in ("AC", "DC")}
print("ARZ tanimlari: %s" % ", ".join(ARZ))


kk = pd.Series(list(zip(hs["ga"], hs["gb"])))
gruplar = kk.groupby(kk).groups
anahtarlar = list(gruplar)
if deneme:
    rng = np.random.default_rng(20260918)
    anahtarlar = [anahtarlar[i] for i in rng.choice(len(anahtarlar), size=min(deneme, len(anahtarlar)), replace=False)]
print("Benzersiz kenar: %d (islenecek %d)" % (len(gruplar), len(anahtarlar)), flush=True)

A = {az: np.zeros(len(K)) for az in ARZ}
for gi, key in enumerate(anahtarlar):
    idx = np.asarray(gruplar[key])
    ga, gb = int(key[0]), int(key[1])
    # kesin arama siniri (revizyon): E = D - phi(o) + phi(g) <= E0  <=>  D <= E0 + phi(o) - phi(g);
    # phi(o) <= PHIMAX oldugundan D <= E0 + PHIMAX - phi(g) hicbir kokeni atlamaz
    lim = E0["DC" if (n_dc[idx] > 0).any() else "AC"] + PHIMAX - min(phi[ga], phi[gb])
    D = dijkstra(Ge_r, directed=True, indices=[ga, gb], limit=lim)
    ulas = np.nonzero(np.isfinite(D[0]) | np.isfinite(D[1]))[0]
    if not len(ulas):
        continue
    say = ptr[ulas + 1] - ptr[ulas]
    ulas = ulas[say > 0]; say = say[say > 0]
    if not len(ulas):
        continue
    oi = np.repeat(ptr[ulas], say) + (np.arange(say.sum()) - np.repeat(np.cumsum(say) - say, say))
    dn = np.repeat(np.arange(len(ulas)), say)
    Ea = D[0][ulas] - phi[ulas] + phi[ga]        # gercek kWh (potansiyel geri alinir)
    Eb = D[1][ulas] - phi[ulas] + phi[gb]
    for s in idx:
        fr = hs["frac"][s]
        ea = Ea + fr * hs["e_kwh"][s] if hs["fwd"][s] else np.full(len(ulas), np.inf)
        eb = Eb + (1 - fr) * hs["e_kwh_r"][s] if hs["rev"][s] else np.full(len(ulas), np.inf)
        d_kwh = np.minimum(ea, eb)[dn] + o_bagkwh[oi] + hs["baglanma_km"][s] * EBAR
        d_kwh = np.maximum(d_kwh, 0.0)
        for tip in ("AC", "DC"):
            if max(ARZ[az][tip][s] for az in ARZ) <= 0:
                continue
            W = W_gauss(d_kwh, E0[tip])
            nz = W > 0
            if not nz.any():
                continue
            ii, w = oi[nz], W[nz]
            talep = np.sum(o_pop[ii] * w)
            if talep <= 0:
                continue
            for az in ARZ:
                Sj = ARZ[az][tip][s]
                if Sj > 0:
                    np.add.at(A[az], ii, (Sj / talep) * w)
    if gi % 250 == 0:
        print("  kenar %d/%d | %.0f sn" % (gi, len(anahtarlar), time.time() - t0), flush=True)

il = pd.read_csv(os.path.join(V, "faz0_ilce.csv")).set_index("shapeID")
Ks = K.iloc[sira].copy()
Ks["sege"] = Ks.shapeID.map(il.sege_skor)
Ks["kademe"] = pd.qcut(Ks.sege.rank(method="first"), 6, labels=False)
cikti = Ks[["shapeID", "il_ad", "ilce_ad", "nufus", "sege"]].copy()
sonuc = []
for az, v in A.items():
    v = v * 1000
    cikti["%s_%s" % (ETIKET, az)] = v
    ilce_A = pd.DataFrame({"A": v, "w": o_pop, "sid": Ks.shapeID.values}).groupby("sid").apply(
        lambda q: np.average(q.A, weights=q.w), include_groups=False)
    ilce_w = pd.DataFrame({"w": o_pop, "sid": Ks.shapeID.values}).groupby("sid").w.sum().reindex(ilce_A.index)
    T, arasi = theil_w(v, o_pop, Ks.kademe.values)
    sonuc.append({"havza": ETIKET, "arz": az,
                  "sifir_erisim_nufus_%": round(100 * o_pop[v == 0].sum() / o_pop.sum(), 2),
                  "gini_koken": round(gini_w(v, o_pop), 3),
                  "gini_ilce": round(gini_w(ilce_A.values, ilce_w.values), 3),
                  "theil": round(T, 3), "theil_SEGE_arasi_%": round(100 * arasi / T, 1),
                  "spearman_SEGE_ilce": round(pd.Series(ilce_A.values, index=ilce_A.index).corr(
                      il.sege_skor.reindex(ilce_A.index), method="spearman"), 2)})
R_ = pd.DataFrame(sonuc)
if not deneme:
    cikti.to_csv(os.path.join(V, (("faz5_enerjik%s_koken.csv" % EKD) if ETIKET=="ENERJIK" else "faz5_enerji_koken.csv")), index_label="o", encoding="utf-8-sig")
    kol = [c for c in cikti.columns if c.startswith(ETIKET + "_")]
    cikti.groupby("shapeID").apply(lambda q: pd.Series({c: np.average(q[c], weights=q.nufus) for c in kol}), include_groups=False) \
         .join(il[["il_ad", "ilce_ad", "nufus", "sege_skor"]]).to_csv(os.path.join(V, (("faz5_enerjik%s_ilce.csv" % EKD) if ETIKET=="ENERJIK" else "faz5_enerji_ilce.csv")), encoding="utf-8-sig")
    R_.to_csv(os.path.join(V, (("faz5_enerjik%s_ozet.csv" % EKD) if ETIKET=="ENERJIK" else "faz5_enerji_ozet.csv")), index=False, encoding="utf-8-sig")
print("\nENERJI HAVZASI (nufus agirlikli):")
print(R_.to_string(index=False))
print("\nToplam sure %.0f sn" % (time.time() - t0))
