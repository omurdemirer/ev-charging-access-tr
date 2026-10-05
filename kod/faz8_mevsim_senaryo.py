"""Faz 8 — AS4 mevsim/donem senaryolari: il bazinda talep carpanlari + mevsimsel yardimci yuk.

MEVSIM ile BUYUMEYI AYIRMA (kritik):
  - `faz2c_il_ay_tahmin.csv` ay-of-yil periyodiktir (2025-07 ile 2026-07 BIREBIR AYNI) -> saf
    MEVSIMSEL SEKIL. Bundan her ay ulusal paya normalize edilerek IL bazinda mekansal indeks alinir.
  - `olcek_oturum` (faz2c param) 1,044 -> 0,787'ye duser; bu MEVSIM DEGIL BUYUMEdir (soket sayisi
    oturumlardan hizli artiyor). Bu, ULUSAL SEVIYE carpani olarak ayri tutulur.
  Ikisi carpilmaz-karistirilmaz: mekansal indeks paya normalize edildigi icin ulusal mevsimselligi
  icermez, seviye carpani da mekansal bilgi icermez.

  indeks_il,s = [ort_{m in s} T[il,m]/SUM_il T[il,m]] / [ort_{tum ay} T[il,m]/SUM_il T[il,m]]
  lambda_senaryo(istasyon) = oturum_ay_kal(taban, 13 ay ortalamasi) * indeks_il,s * seviye_s

YARDIMCI YUK (P_aux) — Fiori, Ahn & Rakha (2016), kaynagin tam metninden ICERIK DOGRULANDI:
  taban 0,700 kW | 25 C: 0,85 kW | 35 C: 1,2 kW | 5 C: 2,2 kW
  -> YAZ senaryolarinda 1,2 kW (35 C), KIS senaryosunda 2,2 kW (5 C).
  ⚠️ Kisin batarya sogugu rejeneratif verimi de dusurur; bunun DOGRULANMIS bir degeri elimizde YOK,
     bu yuzden eta_regen mevsime gore DEGISTIRILMEDI. Kis etkisi bu yonden ALT SINIRDIR.

Senaryolar:
  YAZ2025 : yaz mekansal indeks · seviye(2025-07,08) · P_aux 1,2
  KIS     : kis mekansal indeks · seviye(2025-12,2026-01,02) · P_aux 2,2
  YAZ2026 : yaz mekansal indeks · seviye(2026-07) · P_aux 1,2     [YAZ2025 ile farki = DONEM etkisi]

Cikti: ../veri/faz8_senaryo.json
"""
import os, sys, json
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")

par = json.load(open(os.path.join(V, "faz2c_lambda_param.json"), encoding="utf-8"))
ay, ol = par["aylar"], np.array(par["olcek_oturum"])
T = pd.read_csv(os.path.join(V, "faz2c_il_ay_tahmin.csv")).set_index("il_ad")[ay]
pay = T / T.sum(axis=0)                       # her ayin ULUSAL payina normalize -> saf mekansal sekil
taban = pay.mean(axis=1)

AYLAR = {"YAZ": ["2025-07", "2025-08"], "KIS": ["2025-12", "2026-01", "2026-02"]}
olm = dict(zip(ay, ol)); ort = float(ol.mean())
SEN = {
    "YAZ2025": {"mevsim": "YAZ", "seviye_ay": ["2025-07", "2025-08"], "P_aux_kW": 1.2, "sicaklik_C": 35},
    "KIS":     {"mevsim": "KIS", "seviye_ay": ["2025-12", "2026-01", "2026-02"], "P_aux_kW": 2.2, "sicaklik_C": 5},
    "YAZ2026": {"mevsim": "YAZ", "seviye_ay": ["2026-07"], "P_aux_kW": 1.2, "sicaklik_C": 35},
}
cikti = {"_kaynak": {
    "mekansal_indeks": "faz2c_il_ay_tahmin.csv, her ay ulusal paya normalize (ay-of-yil periyodik, trend icermez)",
    "seviye": "faz2c_lambda_param.olcek_oturum / 13-ay ortalamasi (BUYUME sinyali)",
    "P_aux": "Fiori, Ahn & Rakha (2016) — 0,7 taban / 0,85 (25C) / 1,2 (35C) / 2,2 (5C); kaynagin tam metninden dogrulandi",
    "uyari": "eta_regen mevsime gore DEGISTIRILMEDI (kis batarya sogugu icin dogrulanmis deger yok) -> kis etkisi ALT SINIR"}}
for ad, s in SEN.items():
    idx = (pay[AYLAR[s["mevsim"]]].mean(axis=1) / taban)
    sev = float(np.mean([olm[a] for a in s["seviye_ay"]]) / ort)
    cikti[ad] = {"seviye": sev, "P_aux_kW": s["P_aux_kW"], "sicaklik_C": s["sicaklik_C"],
                 "mevsim": s["mevsim"], "il_indeks": {k: float(v) for k, v in idx.items()}}
    print("%-8s seviye %.4f | P_aux %.2f kW (%d C) | il indeks: min %.3f (%s), max %.3f (%s), ort %.4f"
          % (ad, sev, s["P_aux_kW"], s["sicaklik_C"], idx.min(), idx.idxmin(), idx.max(), idx.idxmax(), idx.mean()))

# --- BLOK KISITINA KALIBRASYON (tasarimda listelenen GOZLENEN hedef) ---
# Modellenen il x ay tahmini, gozlenen mevsimsel yogunlasmayi eksik gosteriyor:
# gozlenen ilk10/kalan71 kapasite yogunlugu orani yazdan kisa 1,31 -> 2,02 (x1,54);
# ham indeks yalnizca x1,14 uretiyor. Indeks gucu gamma ile olceklenip hedefe oturtulur:
#   indeks' = indeks^gamma   (monoton; il siralamasi korunur)
B = pd.read_csv(os.path.join(V, "blok_karsilastirma_ilk10_kalan.csv"))
wv = B.pivot(index="ay", columns="blok", values="kW_yogunluk")
HEDEF = {"YAZ": float((wv.ilk10 / wv.kalan71).loc[AYLAR["YAZ"]].mean()),
         "KIS": float((wv.ilk10 / wv.kalan71).loc[AYLAR["KIS"]].mean())}
sk = pd.read_csv(os.path.join(V, "faz2c_soket_lambda.csv"), usecols=["il_ad", "P_etkin", "oturum_ay_kal"])
ag = sk.groupby("il_ad").agg(kw=("P_etkin", "sum"), lam=("oturum_ay_kal", "sum"))
i10 = set(ag.lam.sort_values(ascending=False).head(10).index)
ag["i10"] = ag.index.isin(i10)


def blok_orani(idx, gamma):
    lam = ag.lam * ag.index.map(idx).astype(float) ** gamma
    ps, ks = lam.groupby(ag.i10).sum(), ag.kw.groupby(ag.i10).sum()
    yog = (ps / ps.sum()) / (ks / ks.sum())
    return float(yog[True] / yog[False])


for ad, s_ in SEN.items():
    idx = cikti[ad]["il_indeks"]
    hedef = HEDEF[s_["mevsim"]]
    gl = np.arange(0.0, 12.01, 0.05)
    vals = np.array([blok_orani(idx, g) for g in gl])
    gam = float(np.interp(hedef, vals, gl)) if vals[-1] >= hedef >= vals[0] else float(gl[int(np.argmin(abs(vals - hedef)))])
    ulas = blok_orani(idx, gam)
    cikti[ad]["gamma"] = gam
    cikti[ad]["blok_orani_hedef"] = hedef
    cikti[ad]["blok_orani_ulasilan"] = ulas
    cikti[ad]["il_indeks_ham"] = dict(idx)
    cikti[ad]["il_indeks"] = {k: float(v) ** gam for k, v in idx.items()}
    print("%-8s blok orani hedef %.3f -> gamma %.2f ile ulasilan %.3f (ham gamma=1: %.3f)"
          % (ad, hedef, gam, ulas, blok_orani(idx, 1.0)))

json.dump(cikti, open(os.path.join(V, "faz8_senaryo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nKaydedildi: faz8_senaryo.json")
print("\nYAZ/KIS mekansal orani en yuksek 5 il:")
r = (pay[AYLAR["YAZ"]].mean(axis=1) / pay[AYLAR["KIS"]].mean(axis=1)).sort_values(ascending=False)
print(r.head(5).round(3).to_string())
print("en dusuk 5 il:")
print(r.tail(5).round(3).to_string())
