"""Faz 2c — ilce talebi + Huff dagitimi ile istasyon talebi (lambda) ve EPDK il hedeflerine kalibrasyon.

Model (ana kara ilcesi d, ay m; halka acik istasyon s; il i)
  Talep:  Dem_dm = w_R*R_d + w_C*C_d*k_C(m) + w_T*T_dm
    R_d  = nufus_d * exp(gamma * SEGE_d)                              (ilce SEGE'si)
    C_d  = KGM il tasit-km payi x il ici ana yol payi (OSM km; motorway 3, trunk 2, primary 1 agirlik)
           k_C(m) = 1 + delta (Tem-Agu), 1 diger; yillik ortalama 1
    T_dm = ilce yerli geceleme payi x 12 x ulusal ay payi x il mevsim duzeltmesi (Tem-Agu: il yaz katsayisi)
  Huff:   P(s|d) = kW_s^alpha * exp(-beta * dk_ds) / sum_s' (...)   (s: <=120 dk halka acik istasyonlar)
  Istasyon talebi: lambda_sm = sum_d Dem_dm * P(s|d);  il payi S_im = 100 * sum_{s in i} lambda_sm / sum lambda
  -> tuketim ISTASYONUN ilinde kaydedilir: mekansal tasma modelde kendiliginden vardir.
Hiz: P yalniz (alpha, beta)'ya baglidir -> ilce->il akis matrisi Q (969x81) onbellege alinir.
Hedef: EPDK ilk-10 il paylari (2025-07 -> 2026-07). Testler: LOPO (il disarida), KALAN TESTI
  (yalniz ilk-10 ile kalibre; gorulmemis illerin toplami / gercek kalan). Olcutler: Faz 2b 1,53 / 1,59x; kW 1,22 / 1,36x.
Nihai: ilk-10 + kalan hedefli kalibrasyon -> istasyon bazinda aylik talep payi ve oturum (EPDK ulusal adet x pay).
Girdi: ../veri/osm/ilce_istasyon_od.csv, ../veri/istasyon_duzeltilmis.csv, ../veri/faz0_ilce.gpkg, ../veri/osm/ag.npz,
       ../veri/kgm/kgm_2025_il_tasit_km.csv, ../veri/turizm/turizm_il.csv, turizm_ilce.csv, ktb_bakanlik_yillik_2025.xlsx,
       ../veri/epdk_panel_il_top10.csv, ../veri/epdk_panel_aylik.csv
Cikti: ../veri/faz2c_param.json, faz2c_il_ay_tahmin.csv, faz2c_istasyon_talep.csv, faz2c_ilce_tasma.csv
"""
import sys, os, re, json, time, unicodedata
import numpy as np, pandas as pd, geopandas as gpd
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
OSM = os.path.join(V, "osm")
YAZ = (7, 8)
ADA = {("ÇANAKKALE", "GÖKÇEADA"), ("ÇANAKKALE", "BOZCAADA"), ("BALIKESİR", "MARMARA"), ("İSTANBUL", "ADALAR")}
AY_NO = {"OCAK": 1, "ŞUBAT": 2, "MART": 3, "NİSAN": 4, "MAYIS": 5, "HAZİRAN": 6, "TEMMUZ": 7,
         "AĞUSTOS": 8, "EYLÜL": 9, "EKİM": 10, "KASIM": 11, "ARALIK": 12}
tr_upper = lambda s: str(s).strip().replace("i", "İ").replace("ı", "I").upper()


def n2(s):
    s = str(s).replace("İ", "i").replace("I", "ı").lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).replace("ı", "i")
    return re.sub(r"[^a-z]", "", s)


t0 = time.time()
# ---------------- ilceler (ana kara) ----------------
g = gpd.read_file(os.path.join(V, "faz0_ilce.gpkg"))[["shapeID", "il_ad", "ilce_ad", "nufus", "sege_skor", "geometry"]].to_crs(4326)
g = g[[(a, b) not in ADA for a, b in zip(g.il_ad, g.ilce_ad)]].reset_index(drop=True)
D = len(g)
dix = dict(zip(g.shapeID, range(D)))
IL = sorted(g.il_ad.unique()); NI = len(IL); iix = {n: i for i, n in enumerate(IL)}
d_il = g.il_ad.map(iix).values

# ---------------- istasyonlar ve OD ----------------
st = pd.read_csv(os.path.join(V, "istasyon_duzeltilmis.csv"))
st = st[(st.hizmet == "HALKA_ACIK") & ~st.ada.astype(bool)].reset_index(drop=True)
six = dict(zip(st.ist_no, range(len(st))))
od = pd.read_csv(os.path.join(OSM, "ilce_istasyon_od.csv"), usecols=["shapeID", "ist_no", "dk"])
od = od[od.dk.notna() & od.ist_no.isin(six) & od.shapeID.isin(dix)]
od = od.assign(d=od.shapeID.map(dix), s=od.ist_no.map(six)).sort_values(["d", "dk"])
d_arr, s_arr, t_arr = od.d.values, od.s.values, od.dk.values
bas = np.r_[0, np.flatnonzero(np.diff(d_arr)) + 1]
say = np.diff(np.r_[bas, len(d_arr)])
assert len(bas) == D and np.array_equal(np.unique(d_arr), np.arange(D)), "her ilcenin en az bir istasyonu olmali"
logA = np.log(np.maximum(st.kw.values, 7.0))[s_arr]
s_il_arr = st.il_ad.map(iix).values[s_arr]
assert not np.isnan(s_il_arr.astype(float)).any()
print("Ilce %d | istasyon %d | OD satiri %d | il %d (%.0f sn)" % (D, len(st), len(od), NI, time.time() - t0), flush=True)

# ---------------- talep bilesenleri ----------------
tkm = pd.read_csv(os.path.join(V, "kgm", "kgm_2025_il_tasit_km.csv")).set_index("il_ad").tkm_toplam
A = np.load(os.path.join(OSM, "ag.npz"))
cw = np.select([A["sinif"] == 1, A["sinif"] == 3, A["sinif"] == 5], [3.0, 2.0, 1.0], 0.0) * A["km"]
k = cw > 0
mlon = (A["n_lon"][A["U"][k]] + A["n_lon"][A["V"][k]]) / 2
mlat = (A["n_lat"][A["U"][k]] + A["n_lat"][A["V"][k]]) / 2
pts = gpd.GeoDataFrame({"w": cw[k]}, geometry=gpd.points_from_xy(mlon, mlat), crs=4326)
yolw = gpd.sjoin(pts, g[["shapeID", "geometry"]], predicate="within").groupby("shapeID").w.sum()
g["yolw"] = g.shapeID.map(yolw).fillna(0.0)
il_yolw = g.groupby("il_ad").yolw.transform("sum")
C = (tkm.reindex(g.il_ad).values / tkm.sum()) * np.where(il_yolw > 0, g.yolw / il_yolw, 0.0)
C = C / C.sum()

tz_il = pd.read_csv(os.path.join(V, "turizm", "turizm_il.csv"), index_col=0)
tz = pd.read_csv(os.path.join(V, "turizm", "turizm_ilce.csv"))
tz["k"] = tz.il_ad + "|" + tz.ilce.map(n2)
g["k"] = g.il_ad + "|" + g.ilce_ad.map(n2)
eslesen = tz.k.isin(set(g.k))
Tv = g.k.map(tz[eslesen].groupby("k").gec_yerli_2025tah.sum()).fillna(0.0).values
art = tz[~eslesen].groupby("il_ad").gec_yerli_2025tah.sum()          # eslesmeyen: il icinde nufusa gore
pay_nufus = g.nufus / g.groupby("il_ad").nufus.transform("sum")
Tv = Tv + g.il_ad.map(art).fillna(0.0).values * pay_nufus.values
print("Turizm ilce eslesmesi: %d/%d satir; eslesmeyen yerli geceleme payi %%%.1f (il icinde nufusa gore dagitildi)"
      % (eslesen.sum(), len(tz), 100 * tz[~eslesen].gec_yerli_2025tah.sum() / tz.gec_yerli_2025tah.sum()), flush=True)
Tsh = Tv / Tv.sum()

a = pd.read_excel(os.path.join(V, "turizm", "ktb_bakanlik_yillik_2025.xlsx"), sheet_name="Ay", header=None, skiprows=3)
a = a[a.iloc[:, 0].map(tr_upper).isin(AY_NO)]
wT = pd.Series(pd.to_numeric(a.iloc[:, 5]).values, index=a.iloc[:, 0].map(tr_upper).map(AY_NO)).sort_index()
wT = wT / wT.sum()
yk = tz_il.yaz_katsayisi.replace([np.inf, -np.inf], np.nan).fillna(1.0).reindex(g.il_ad).fillna(1.0).values
sw = wT[7] + wT[8]
q = np.clip((1 - yk * sw) / (1 - sw), 0, None)

# ---------------- hedefler ----------------
t = pd.read_csv(os.path.join(V, "epdk_panel_il_top10.csv"))
aylar = sorted(t.ay.unique()); M = len(aylar)
mi = np.array([int(x[-2:]) for x in aylar]); yaz = np.isin(mi, YAZ)
Pobs = np.full((NI, M), np.nan)
for r in t.itertuples():
    Pobs[iix[r.il], aylar.index(r.ay)] = r.pay
OBS = ~np.isnan(Pobs)
kalan = 100 - np.nansum(Pobs, axis=0)
T0 = Tsh[:, None] * 12 * wT.loc[mi].values[None, :] * np.where(yaz[None, :], yk[:, None], q[:, None])
pop, sege = g.nufus.values.astype(float), g.sege_skor.values

# ---------------- model ----------------
_q = {"key": None, "Q": None}


def Qmat(alpha, beta):
    key = (round(alpha, 12), round(beta, 12))
    if _q["key"] == key:
        return _q["Q"]
    x = alpha * logA - beta * t_arr
    x = x - np.repeat(np.maximum.reduceat(x, bas), say)
    w = np.exp(x)
    p = w / np.repeat(np.add.reduceat(w, bas), say)
    Q = np.bincount(d_arr * NI + s_il_arr, weights=p, minlength=D * NI).reshape(D, NI)
    _q.update(key=key, Q=Q, p=p)
    return Q


def talep(th, aktif, delta_on):
    gamma, uC, uT, delta = th[:4]
    R = pop * np.exp(gamma * sege); R = R / R.sum()
    kC = (1 + delta * yaz) / (1 + delta * 2 / 12) if delta_on else np.ones(M)
    lg = np.array([0.0, uC if aktif[1] else -np.inf, uT if aktif[2] else -np.inf])
    w = np.exp(lg - lg.max()); w = w / w.sum()
    return w[0] * R[:, None] + w[1] * C[:, None] * kC[None, :] + w[2] * T0, w


def il_pay(th, aktif, delta_on):
    Dem, w = talep(th, aktif, delta_on)
    Z = Qmat(th[4], np.exp(th[5])).T @ Dem
    return 100 * Z / Z.sum(axis=0, keepdims=True), w


def kayip(th, aktif, delta_on, mask, kalan_dahil):
    S, _ = il_pay(th, aktif, delta_on)
    L = np.sum((np.log(Pobs[mask]) - np.log(S[mask])) ** 2)
    if kalan_dahil:
        L += np.sum((np.log(kalan) - np.log(np.where(OBS, 0, S).sum(axis=0))) ** 2)
    return L


SINIR = [(-1, 3), (-8, 8), (-8, 8), (0, 3), (0, 3), (np.log(0.005), np.log(0.5))]


def kalibre(aktif, delta_on, mask=OBS, kalan_dahil=False, baslangic=None):
    free = [0, 4, 5] + [kk for kk, on in ((1, aktif[1]), (2, aktif[2]), (3, delta_on)) if on]
    adaylar = baslangic if baslangic is not None else [
        np.array([g0, 0.0, 0.0, 0.3, a0, np.log(b0)]) for g0 in (0.0, 1.0) for a0 in (0.5, 1.5) for b0 in (0.03, 0.1)]
    best = None
    for x0 in adaylar:
        def f(x):
            th = np.array(x0, dtype=float); th[free] = x
            return kayip(th, aktif, delta_on, mask, kalan_dahil)
        r = minimize(f, np.asarray(x0)[free], bounds=[SINIR[kk] for kk in free], method="L-BFGS-B")
        if best is None or r.fun < best[1]:
            th = np.array(x0, dtype=float); th[free] = r.x
            best = (th, r.fun)
    return best


MODELLER = {"ikamet": ((1, 0, 0), False), "uc bilesen + koridor yaz": ((1, 1, 1), True)}
prov = [i for i in range(NI) if OBS[i].any()]
sonuc, ic = [], {}
for isim, (aktif, dl) in MODELLER.items():
    t1 = time.time()
    th, _ = kalibre(aktif, dl)
    S, w = il_pay(th, aktif, dl)
    e = np.log(Pobs[OBS]) - np.log(S[OBS])
    lopo = []
    for pi in prov:
        m2 = OBS.copy(); m2[pi] = False
        th2, _ = kalibre(aktif, dl, mask=m2, baslangic=[th])
        S2, _ = il_pay(th2, aktif, dl)
        lopo += list(np.log(Pobs[pi, OBS[pi]]) - np.log(S2[pi, OBS[pi]]))
    oran = np.where(OBS, 0, S).sum(axis=0) / kalan
    ic[isim] = (th, S, w)
    sonuc.append({"model": isim, "gamma": round(th[0], 2), "alpha": round(th[4], 2), "beta_1/dk": round(np.exp(th[5]), 4),
                  "delta": round(th[3], 2) if dl else "-", "w_ikamet": round(w[0], 3), "w_koridor": round(w[1], 3),
                  "w_turizm": round(w[2], 3), "ic_carpan": round(np.exp(np.sqrt(np.mean(e ** 2))), 2),
                  "LOPO_carpan": round(np.exp(np.sqrt(np.mean(np.square(lopo)))), 2),
                  "kalan_ort": round(oran.mean(), 2), "kalan_min": round(oran.min(), 2), "kalan_max": round(oran.max(), 2)})
    print("  tamam: %s (%.0f sn)" % (isim, time.time() - t1), flush=True)

R_ = pd.DataFrame(sonuc)
print("\nKALIBRASYON (yalniz EPDK ilk-10 ile; %d gozlem, %d il, %d ay)" % (OBS.sum(), len(prov), M))
print(R_.to_string(index=False))
print("Olcutler: Faz 2b (il duzeyi) kalan 1,53 / LOPO 1,59x · yalniz kW kalan 1,22 / LOPO 1,36x")

th, S, w = ic["uc bilesen + koridor yaz"]
print("\nYaz payi / diger aylar (gozlenen | model):")
for n in ["BALIKESİR", "MUĞLA", "KONYA", "ANTALYA", "İZMİR", "BOLU", "BURSA", "KOCAELİ", "İSTANBUL", "ANKARA"]:
    i = iix[n]; o = OBS[i]
    if (o & yaz).any() and (o & ~yaz).any():
        print("  %-10s %.2f | %.2f" % (n, Pobs[i, o & yaz].mean() / Pobs[i, o & ~yaz].mean(), S[i, yaz].mean() / S[i, ~yaz].mean()))

# ---------------- nihai model ----------------
aktif, dl = MODELLER["uc bilesen + koridor yaz"]
thF, _ = kalibre(aktif, dl, kalan_dahil=True, baslangic=[th])
SF, wF = il_pay(thF, aktif, dl)
DemF, _ = talep(thF, aktif, dl)
Qmat(thF[4], np.exp(thF[5])); pF = _q["p"]
lam = np.vstack([np.bincount(s_arr, weights=DemF[d_arr, j] * pF, minlength=len(st)) for j in range(M)]).T
lam = lam / lam.sum(axis=0, keepdims=True)
adet = pd.read_csv(os.path.join(V, "epdk_panel_aylik.csv")).set_index("ay").adet.reindex(aylar).values
out = st[["ist_no", "il_ad", "ilce_ad", "soket", "dc", "kw", "lat", "lon"]].copy()
for j, ay in enumerate(aylar):
    out["pay_" + ay] = lam[:, j]
    out["oturum_" + ay] = lam[:, j] * adet[j]
out["oturum_ay_ort"] = out[[c for c in out if c.startswith("oturum_")]].mean(axis=1)
out.to_csv(os.path.join(V, "faz2c_istasyon_talep.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(SF, index=IL, columns=aylar).rename_axis("il_ad").round(4).to_csv(os.path.join(V, "faz2c_il_ay_tahmin.csv"), encoding="utf-8-sig")

Qf = _q["Q"]
tas = pd.DataFrame({"il_ad": g.il_ad, "ilce_ad": g.ilce_ad, "ic_il_payi": Qf[np.arange(D), d_il]})
tas.to_csv(os.path.join(V, "faz2c_ilce_tasma.csv"), index=False, encoding="utf-8-sig")
DemAvg = DemF.mean(axis=1)
il_ic = pd.Series(DemAvg * Qf[np.arange(D), d_il], index=g.il_ad).groupby(level=0).sum() / pd.Series(DemAvg, index=g.il_ad).groupby(level=0).sum()
json.dump({"model": "uc bilesen + koridor yaz (ilk-10 + kalan hedefli)", "gamma": thF[0], "delta": thF[3], "alpha": thF[4],
           "beta_1_dk": float(np.exp(thF[5])), "w_ikamet": wF[0], "w_koridor": wF[1], "w_turizm": wF[2], "aylar": aylar},
          open(os.path.join(V, "faz2c_param.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\nNIHAI (ilk-10 + kalan hedefli): gamma=%.2f alpha=%.2f beta=%.4f/dk delta=%.2f | w ikamet=%.3f koridor=%.3f turizm=%.3f"
      % (thF[0], thF[4], np.exp(thF[5]), thF[3], *wF))
print("Kalan orani (nihai):", np.round(np.where(OBS, 0, SF).sum(axis=0) / kalan, 2).tolist())
print("\nTasma tanisi — ilin kendi talebinin kendi ilinde karsilanan payi (dusuk = komsu ile gidiyor):")
print((100 * il_ic).sort_values().head(10).round(1).to_string())
print("  ... en yuksek: ", (100 * il_ic).sort_values().tail(4).round(1).to_dict())
print("\nEn yuksek aylik ortalama oturum tahmini, 12 istasyon:")
print(out.sort_values("oturum_ay_ort", ascending=False)[["ist_no", "il_ad", "ilce_ad", "soket", "dc", "kw", "oturum_ay_ort"]].head(12).round(0).to_string(index=False))
print("Toplam sure: %.0f sn" % (time.time() - t0))
