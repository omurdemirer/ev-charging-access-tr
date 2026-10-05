"""Faz 3 — erisim/esitlik tarafi: istasyon-merkezli TEK GECISLI E2SFCA (v3, 18.09.2026)

NEDEN: Ture ozgu havza (DC 40 km) ile koken-merkezli OD matrisi ~450 milyon satira ve ~15 saate cikiyordu.
COZUM: Ciftleri SAKLAMADAN, TERS graf uzerinde istasyondan geriye Dijkstra ile tek gecis:
  istasyon j: d_j(o) -> W -> talep_j = sum_o P_o W(d)  ->  R_j = S_j / talep_j  ->  A_o += R_j W(d)
2SFCA'nin iki adimi ayni satirda tamamlanir; disk ihtiyaci sifir, bellek birkac MB.
Ters graf: G^T uzerinde istasyon dugumunden hesaplanan mesafe = kokenden istasyona mesafe (yon korunur).

HAVZA (karar 18.09.2026, birincil kanit: Nicholas, Tal & Turrentine 2020, CARB 12-319):
  DCFC oturumlarinin %65'i evden 25 mil icinde; L2+DCFC'nin yalnizca %25'i 5 mil icinde; DCFC'nin %97'si >1 mil.
  -> BIRINCIL: AC Gaussian 5 mil (8,05 km), DC Gaussian 25 mil (40,23 km)
  -> Duyarlilik: AC 15 mil · ustel beta=0,08/DAKIKA (Levinson & Kumar 1995; Gazmeh+2024 bunu km'ye uygulamis — hata)
     · 15 dk esik (Carlton & Sultana 2022) · 3 mil (Yu+2025)  [son ikisi literaturle karsilastirilabilirlik icin]
ARZ: NOM (soket sayisi) · K1 (saatlik oturum kapasitesi) · K1g (arac kabul sinirli etkin kW) — AC ve DC ayri.
Girdi : ../veri/osm/ag*.npz, ../veri/osm/izgara_koken.csv (faz1c ile uretilen kokenler), ../veri/istasyon_duzeltilmis.csv,
        ../veri/faz2c_soket_lambda.csv, ../veri/faz2c_lambda_param.json, ../veri/faz0_ilce.csv
Cikti : ../veri/faz3_erisim_koken.csv, faz3_erisim_ilce.csv, faz3_esitsizlik_ozet.csv
Kullanim: python faz3_erisim_hesapla.py [--deneme 200]
"""
import sys, os, time, json
import numpy as np, pandas as pd
from scipy.sparse.csgraph import dijkstra
from ag_ortak import Ag

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
OSM = os.path.join(V, "osm")
MIL, BAG_HIZ = 1.609344, 30.0
LIM_KM_DC, LIM_KM_AC, LIM_DK = 25 * MIL + 0.1, 15 * MIL + 0.1, 45.0
HAVZA = {  # havza adi -> tur -> (fonksiyon, parametre, olcu)
    "BIRINCIL": {"AC": ("gauss", 5 * MIL, "km"), "DC": ("gauss", 25 * MIL, "km")},
    "AC15mil": {"AC": ("gauss", 15 * MIL, "km"), "DC": ("gauss", 25 * MIL, "km")},
    "USTEL_dk": {"AC": ("ustel", 0.08, "dk"), "DC": ("ustel", 0.08, "dk")},
    "T15dk": {"AC": ("esik", 15.0, "dk"), "DC": ("esik", 15.0, "dk")},
    "3mil": {"AC": ("gauss", 3 * MIL, "km"), "DC": ("gauss", 3 * MIL, "km")},
}
deneme = int(sys.argv[sys.argv.index("--deneme") + 1]) if "--deneme" in sys.argv else 0
t0 = time.time()


def W_hesapla(tanim, d):
    tur, p, _ = tanim
    if tur == "gauss":
        return np.where(d <= p, (np.exp(-d * d / (2 * p * p)) - np.exp(-0.5)) / (1 - np.exp(-0.5)), 0.0)
    if tur == "esik":
        return np.where(d <= p, 1.0, 0.0)
    return np.exp(-p * d)


def gini_w(x, w):
    o = np.argsort(x); x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    if (x * w).sum() == 0:
        return np.nan
    cw = np.r_[0, np.cumsum(w) / w.sum()]; cx = np.r_[0, np.cumsum(x * w) / (x * w).sum()]
    return 1 - np.sum((cw[1:] - cw[:-1]) * (cx[1:] + cx[:-1]))


def theil_w(x, w, grup):
    x = np.maximum(np.asarray(x, float), 1e-12); w = np.asarray(w, float) / np.sum(w)
    mu = np.sum(w * x); T = np.sum(w * x / mu * np.log(x / mu))
    gk = pd.DataFrame({"x": x, "w": w, "g": grup}).groupby("g").apply(
        lambda q: pd.Series({"w": q.w.sum(), "mu": np.sum(q.w * q.x) / q.w.sum()}), include_groups=False)
    return T, np.sum(gk.w * gk.mu / mu * np.log(gk.mu / mu))


# --- kokenler ve ag ---
K = pd.read_csv(os.path.join(OSM, "izgara_koken.csv"), index_col="o")
ag = Ag(OSM)
sira = np.argsort(K.dugum.values, kind="stable")
o_dugum = K.dugum.values[sira].astype(np.int64)
o_pop = K.nufus.values[sira]
o_bagkm = K.baglanma_km.values[sira]
o_bagdk = o_bagkm / BAG_HIZ * 60
ptr = np.searchsorted(o_dugum, np.arange(ag.N + 1))       # dugum -> koken dilimi
print("Koken %d | dugum %d | nufus %.0f | ag yuklendi %.0f sn" % (len(K), ag.N, o_pop.sum(), time.time() - t0), flush=True)
Gt_r, Gk_r = ag.Gt.T.tocsr(), ag.Gk.T.tocsr()

# --- istasyonlar ve arz ---
S = pd.read_csv(os.path.join(V, "istasyon_duzeltilmis.csv"))
S = S[(S.hizmet == "HALKA_ACIK") & ~S.ada.astype(bool)].reset_index(drop=True)
hs = ag.kenara_bagla(S.lon.values, S.lat.values)
par = json.load(open(os.path.join(V, "faz2c_lambda_param.json"), encoding="utf-8"))
sk = pd.read_csv(os.path.join(V, "faz2c_soket_lambda.csv"), usecols=["ist_no", "tip", "P_etkin"])
gg = sk.groupby(["ist_no", "tip"]).agg(n=("P_etkin", "size"), kw=("P_etkin", "sum")).unstack(fill_value=0)
n_ac = gg[("n", "AC")].reindex(S.ist_no).fillna(0).values; n_dc = gg[("n", "DC")].reindex(S.ist_no).fillna(0).values
kw_ac = gg[("kw", "AC")].reindex(S.ist_no).fillna(0).values; kw_dc = gg[("kw", "DC")].reindex(S.ist_no).fillna(0).values
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
# revizyon: (30.09.2026): X42=1 ise koordinati duzeltilip ilce merkezine tasinan istasyonlar arzdan
# cikarilir (arz sifir); yalniz NOM/K1 ve cikarilmis surumleri hesaplanir, cikti "_x42" ekiyle yazilir
X42 = os.environ.get("X42") == "1"
if X42:
    _tas = S.koordinat_duzeltildi.astype(bool).values
    print("Tasinan istasyon: %d" % _tas.sum())
    ARZ = {"NOM": ARZ["NOM"], "K1": ARZ["K1"],
           "NOMx42": {t: np.where(_tas, 0.0, ARZ["NOM"][t]) for t in ("AC", "DC")},
           "K1x42": {t: np.where(_tas, 0.0, ARZ["K1"][t]) for t in ("AC", "DC")}}
print("ARZ tanimlari: %s" % ", ".join(ARZ))


# ayni kenardaki istasyonlar birlikte islenir (Dijkstra bir kez)
kenar_key = pd.Series(list(zip(hs["ga"], hs["gb"])))
gruplar = kenar_key.groupby(kenar_key).groups
anahtarlar = list(gruplar)
if deneme:
    rng = np.random.default_rng(20260918)
    anahtarlar = [anahtarlar[i] for i in rng.choice(len(anahtarlar), size=min(deneme, len(anahtarlar)), replace=False)]
print("Benzersiz kenar: %d (deneme: %d)" % (len(gruplar), len(anahtarlar)), flush=True)

A = {(hv, az): np.zeros(len(K)) for hv in HAVZA for az in ARZ}
talep_kayit = []
for gi, key in enumerate(anahtarlar):
    idx = np.asarray(gruplar[key])
    ga, gb = int(key[0]), int(key[1])
    dc_var = bool((n_dc[idx] > 0).any())
    lim_km = LIM_KM_DC if dc_var else LIM_KM_AC
    Dk = dijkstra(Gk_r, directed=True, indices=[ga, gb], limit=lim_km)
    Dt = dijkstra(Gt_r, directed=True, indices=[ga, gb], limit=LIM_DK)
    ulas = np.nonzero(np.isfinite(Dk[0]) | np.isfinite(Dk[1]) | np.isfinite(Dt[0]) | np.isfinite(Dt[1]))[0]
    if not len(ulas):
        continue
    say = ptr[ulas + 1] - ptr[ulas]
    ulas = ulas[say > 0]; say = say[say > 0]
    if not len(ulas):
        continue
    oi = np.repeat(ptr[ulas], say) + (np.arange(say.sum()) - np.repeat(np.cumsum(say) - say, say))
    dnode = np.repeat(np.arange(len(ulas)), say)
    for s in idx:                                   # ayni kenardaki her istasyon
        fr, ew_km, ew_dk = hs["frac"][s], hs["e_km"][s], hs["e_dk"][s]
        ka = Dk[0][ulas] + fr * ew_km if hs["fwd"][s] else np.full(len(ulas), np.inf)
        kb = Dk[1][ulas] + (1 - fr) * ew_km if hs["rev"][s] else np.full(len(ulas), np.inf)
        ta = Dt[0][ulas] + fr * ew_dk if hs["fwd"][s] else np.full(len(ulas), np.inf)
        tb = Dt[1][ulas] + (1 - fr) * ew_dk if hs["rev"][s] else np.full(len(ulas), np.inf)
        d_km = np.minimum(ka, kb)[dnode] + o_bagkm[oi] + hs["baglanma_km"][s]
        d_dk = np.minimum(ta, tb)[dnode] + o_bagdk[oi] + hs["baglanma_km"][s] / BAG_HIZ * 60
        # Agirlik ve talep yalnizca (havza, tur)'e baglidir; arz tanimi sadece bir carpandir -> bir kez hesapla.
        for hv, tanim in HAVZA.items():
            for tip in ("AC", "DC"):
                if max(ARZ[az][tip][s] for az in ARZ) <= 0:
                    continue
                d = d_km if tanim[tip][2] == "km" else d_dk
                W = W_hesapla(tanim[tip], d)
                nz = W > 0
                if not nz.any():
                    continue
                idx, w = oi[nz], W[nz]
                talep = np.sum(o_pop[idx] * w)
                if talep <= 0:
                    continue
                talep_kayit.append((hv, tip, talep))
                for az in ARZ:
                    Sj = ARZ[az][tip][s]
                    if Sj > 0:
                        np.add.at(A[(hv, az)], idx, (Sj / talep) * w)
    if gi % 500 == 0:
        print("  kenar %d/%d | %.0f sn" % (gi, len(anahtarlar), time.time() - t0), flush=True)

# --- ozet ---
il = pd.read_csv(os.path.join(V, "faz0_ilce.csv")).set_index("shapeID")
Ks = K.iloc[sira].copy()
Ks["sege"] = Ks.shapeID.map(il.sege_skor)
Ks["kademe"] = pd.qcut(Ks.sege.rank(method="first"), 6, labels=False)
cikti = Ks[["shapeID", "il_ad", "ilce_ad", "nufus", "sege"]].copy()
sonuc = []
for (hv, az), v in A.items():
    v = v * 1000
    cikti["%s_%s" % (hv, az)] = v
    ilce_A = pd.DataFrame({"A": v, "w": o_pop, "sid": Ks.shapeID.values}).groupby("sid").apply(
        lambda q: np.average(q.A, weights=q.w), include_groups=False)
    ilce_w = pd.Series(o_pop, index=Ks.shapeID.values).groupby(level=0).sum().reindex(ilce_A.index)
    T, arasi = theil_w(v, o_pop, Ks.kademe.values)
    sonuc.append({"havza": hv, "arz": az, "sifir_erisim_nufus_%": round(100 * o_pop[v == 0].sum() / o_pop.sum(), 2),
                  "gini_koken": round(gini_w(v, o_pop), 3), "gini_ilce": round(gini_w(ilce_A.values, ilce_w.values), 3),
                  "theil": round(T, 3), "theil_SEGE_arasi_%": round(100 * arasi / T, 1),
                  "spearman_SEGE_ilce": round(pd.Series(ilce_A.values, index=ilce_A.index).corr(il.sege_skor.reindex(ilce_A.index), method="spearman"), 2)})
R_ = pd.DataFrame(sonuc).sort_values(["havza", "arz"])
if not deneme:
    cikti.to_csv(os.path.join(V, "faz3_erisim_koken%s.csv" % ("_x42" if X42 else "")), index_label="o", encoding="utf-8-sig")
    kol = [c for c in cikti.columns if c.split("_")[0] in HAVZA]
    cikti.groupby("shapeID").apply(lambda q: pd.Series({c: np.average(q[c], weights=q.nufus) for c in kol}), include_groups=False) \
         .join(il[["il_ad", "ilce_ad", "nufus", "sege_skor"]]).to_csv(os.path.join(V, "faz3_erisim_ilce%s.csv" % ("_x42" if X42 else "")), encoding="utf-8-sig")
    R_.to_csv(os.path.join(V, "faz3_esitsizlik_ozet%s.csv" % ("_x42" if X42 else "")), index=False, encoding="utf-8-sig")
print("\nESITSIZLIK OZETI%s (nufus agirlikli):" % (" [DENEME]" if deneme else ""))
print(R_.to_string(index=False))
print("\nToplam sure %.0f sn" % (time.time() - t0))
