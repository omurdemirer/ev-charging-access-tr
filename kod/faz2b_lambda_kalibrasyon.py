"""Faz 2b — il duzeyinde aylik sarj talebi (lambda) modeli ve kalibrasyonu.

Model (il i, ay m):
  D_im = w_R * R_i + w_C * C_i * k_C(m) + w_T * T_i(m)
  R_i    ikamet : nufus_i * exp(gamma * SEGE_i), paya normalize (gamma = gelismislik esnekligi)
  C_i    koridor: KGM 2025 yillik tasit-km payi;  k_C(m) = 1 + delta (Tem-Agu), 1 (diger), yillik ort. = 1
  T_i(m) turizm : yerli geceleme payi x ulusal aylik yerli geceleme endeksi (KTB 2025) x il mevsim duzeltmesi
                  (Tem-Agu: il yaz katsayisi; diger aylar yillik toplam korunacak sekilde olceklenir)
  Tahmin payi S_im = 100 * D_im / sum_j D_jm ;  w_R + w_C + w_T = 1 (ortalama bir ayda bilesen paylari)
Hedef: EPDK ilk-10 il tuketim paylari (2025-07 -> 2026-07), log hata karesi.
Testler: LOPO (il disarida) + KALAN TESTI (ilk-10 ile kalibre, gorulmemis illerin toplami vs gercek kalan).
  Karsilastirma olcutleri (kod/il_tuketim_tahmin.py): yalniz nufus kalan=3,84 LOPO=2,04x · yalniz kW 1,22 / 1,36x
Nihai model: ilk-10 + her ayin kalan toplami birlikte hedeflenerek yeniden kalibre edilir.
Cikti: ../veri/faz2b_lambda_il_ay.csv (81 il x ay tahmini pay), ../veri/faz2b_lambda_param.json
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
YAZ = (7, 8)
AY_NO = {"OCAK": 1, "ŞUBAT": 2, "MART": 3, "NİSAN": 4, "MAYIS": 5, "HAZİRAN": 6, "TEMMUZ": 7,
         "AĞUSTOS": 8, "EYLÜL": 9, "EKİM": 10, "KASIM": 11, "ARALIK": 12}
tr_upper = lambda s: str(s).strip().replace("i", "İ").replace("ı", "I").upper()

# ---------------- veri ----------------
g = pd.read_csv(os.path.join(V, "faz0_ilce.csv"))
g["w"] = g.sege_skor * g.nufus
il = g.groupby("il_ad").agg(nufus=("nufus", "sum"), w=("w", "sum")).sort_index()
il["sege"] = il.w / il.nufus
il["tkm"] = pd.read_csv(os.path.join(V, "kgm", "kgm_2025_il_tasit_km.csv")).set_index("il_ad").tkm_toplam
tz = pd.read_csv(os.path.join(V, "turizm", "turizm_il.csv"), index_col=0)
il["tur"] = tz.gec_yerli_2025tah
il["yaz_k"] = tz.yaz_katsayisi.replace([np.inf, -np.inf], np.nan).fillna(1.0)
assert il[["tkm", "tur", "sege"]].notna().all().all() and len(il) == 81

a = pd.read_excel(os.path.join(V, "turizm", "ktb_bakanlik_yillik_2025.xlsx"), sheet_name="Ay", header=None, skiprows=3)
a = a[a.iloc[:, 0].map(tr_upper).isin(AY_NO)]
wT = pd.Series(pd.to_numeric(a.iloc[:, 5]).values, index=a.iloc[:, 0].map(tr_upper).map(AY_NO)).sort_index()
assert len(wT) == 12, wT
wT = wT / wT.sum()                                   # ulusal yerli geceleme ay payi

t = pd.read_csv(os.path.join(V, "epdk_panel_il_top10.csv"))
aylar = sorted(t.ay.unique()); M = len(aylar)
mi = np.array([int(x[-2:]) for x in aylar]); yaz = np.isin(mi, YAZ)
ad = list(il.index); N = len(ad); ix = {n: i for i, n in enumerate(ad)}
P = np.full((N, M), np.nan)
for r in t.itertuples():
    P[ix[r.il], aylar.index(r.ay)] = r.pay
OBS = ~np.isnan(P)
kalan = 100 - np.nansum(P, axis=0)

pop, sege = il.nufus.values, il.sege.values
C0 = il.tkm.values / il.tkm.sum()
Tsh = il.tur.values / il.tur.sum()
sw = wT[7] + wT[8]
yk = il.yaz_k.values
q = np.clip((1 - yk * sw) / (1 - sw), 0, None)       # yaz disi aylar icin olcek (yillik toplam korunur)
T0 = Tsh[:, None] * 12 * wT.loc[mi].values[None, :] * np.where(yaz[None, :], yk[:, None], q[:, None])


def paylar(th, aktif, delta_on):
    gamma, uC, uT, delta = th
    R = pop * np.exp(gamma * sege); R = R / R.sum()
    kC = (1 + delta * yaz) / (1 + delta * 2 / 12) if delta_on else np.ones(M)
    lg = np.array([0.0, uC if aktif[1] else -np.inf, uT if aktif[2] else -np.inf])
    w = np.exp(lg - lg.max()); w = w / w.sum()
    D = w[0] * R[:, None] + w[1] * C0[:, None] * kC[None, :] + w[2] * T0
    return 100 * D / D.sum(axis=0, keepdims=True), w


def kayip(th, aktif, delta_on, mask, kalan_dahil):
    S, _ = paylar(th, aktif, delta_on)
    L = np.sum((np.log(P[mask]) - np.log(S[mask])) ** 2)
    if kalan_dahil:
        rest = np.where(OBS, 0, S).sum(axis=0)
        L += np.sum((np.log(kalan) - np.log(rest)) ** 2)
    return L


SINIR = [(-2, 3), (-8, 8), (-8, 8), (0, 3)]


def kalibre(aktif, delta_on, mask=OBS, kalan_dahil=False):
    free = [0] + [k for k, on in ((1, aktif[1]), (2, aktif[2]), (3, delta_on)) if on]
    best = None
    for g0 in (-0.5, 0.0, 0.5, 1.0, 1.5):
        for u0 in (-2.0, 0.0, 2.0):
            x0 = np.array([g0, u0, u0, 0.3])[free]
            def f(x):
                th = np.zeros(4); th[free] = x
                return kayip(th, aktif, delta_on, mask, kalan_dahil)
            r = minimize(f, x0, bounds=[SINIR[k] for k in free], method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
    th = np.zeros(4); th[free] = best.x
    return th, best.fun


MODELLER = {  # (ikamet, koridor, turizm), koridor yaz katsayisi
    "ikamet": ((1, 0, 0), False),
    "ikamet+koridor": ((1, 1, 0), False),
    "ikamet+turizm": ((1, 0, 1), False),
    "uc bilesen": ((1, 1, 1), False),
    "uc bilesen + koridor yaz": ((1, 1, 1), True),
}
prov = [i for i in range(N) if OBS[i].any()]
sonuc, ic = [], {}
for isim, (aktif, dl) in MODELLER.items():
    th, L = kalibre(aktif, dl)
    S, w = paylar(th, aktif, dl)
    e = np.log(P[OBS]) - np.log(S[OBS])
    lopo = []
    for p in prov:
        m2 = OBS.copy(); m2[p] = False
        th2, _ = kalibre(aktif, dl, mask=m2)
        S2, _ = paylar(th2, aktif, dl)
        lopo += list(np.log(P[p, OBS[p]]) - np.log(S2[p, OBS[p]]))
    oran = np.where(OBS, 0, S).sum(axis=0) / kalan
    ic[isim] = (th, S, w)
    sonuc.append({"model": isim, "gamma": round(th[0], 2), "delta": round(th[3], 2) if dl else "-",
                  "w_ikamet": round(w[0], 3), "w_koridor": round(w[1], 3), "w_turizm": round(w[2], 3),
                  "ic_carpan": round(np.exp(np.sqrt(np.mean(e ** 2))), 2),
                  "LOPO_carpan": round(np.exp(np.sqrt(np.mean(np.square(lopo)))), 2),
                  "kalan_ort": round(oran.mean(), 2), "kalan_min": round(oran.min(), 2), "kalan_max": round(oran.max(), 2)})
    print("  tamam:", isim, flush=True)

R = pd.DataFrame(sonuc)
print("\nKALIBRASYON (hedef: EPDK ilk-10, %d gozlem, %d il, %d ay) — yalniz ilk-10 ile kalibre" % (OBS.sum(), len(prov), M))
print(R.to_string(index=False))
print("Olcutler: yalniz nufus kalan 3,84 / LOPO 2,04x · yalniz kW kalan 1,22 / LOPO 1,36x"
      "  (kalan = 1,00 ise gorulmemis illerin toplami dogru)")

# yaz orani yeniden uretimi (tam model)
th, S, w = ic["uc bilesen + koridor yaz"]
print("\nYaz payi / diger aylar — gozlenen (EPDK) vs tam model:")
for n in ["BALIKESİR", "MUĞLA", "KONYA", "ANTALYA", "İZMİR", "BOLU", "BURSA", "KOCAELİ", "İSTANBUL", "ANKARA"]:
    i = ix[n]; o = OBS[i]
    if (o & yaz).any() and (o & ~yaz).any():
        g_ = P[i, o & yaz].mean() / P[i, o & ~yaz].mean()
        m_ = S[i, yaz].mean() / S[i, ~yaz].mean()
        print("  %-10s gozlenen %.2f | model %.2f" % (n, g_, m_))

# nihai model: ilk-10 + kalan toplamlari birlikte
aktif, dl = MODELLER["uc bilesen + koridor yaz"]
thF, _ = kalibre(aktif, dl, kalan_dahil=True)
SF, wF = paylar(thF, aktif, dl)
out = pd.DataFrame(SF, index=ad, columns=aylar)
out.index.name = "il_ad"
out.round(4).to_csv(os.path.join(V, "faz2b_lambda_il_ay.csv"), encoding="utf-8-sig")
json.dump({"model": "uc bilesen + koridor yaz (ilk-10 + kalan hedefli)", "gamma": thF[0], "delta": thF[3],
           "w_ikamet": wF[0], "w_koridor": wF[1], "w_turizm": wF[2], "aylar": aylar},
          open(os.path.join(V, "faz2b_lambda_param.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nNIHAI (ilk-10 + kalan hedefli): gamma=%.2f delta=%.2f | w ikamet=%.3f koridor=%.3f turizm=%.3f"
      % (thF[0], thF[3], *wF))
print("Kalan orani (nihai):", np.round(np.where(OBS, 0, SF).sum(axis=0) / kalan, 2).tolist())
