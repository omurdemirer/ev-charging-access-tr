"""Faz 2c (tikaniklik tarafi) — arz temelli, gozlenen tuketime kalibre istasyon/soket talebi (lambda). v3 (16.09.2026)

Kararlar (16.09.2026): (1) Huff tanimlanamadi -> tikaniklik tarafi arz temelli; (2) il ici heterojenlik
STOKASTIK: hangi istasyonun sicak oldugu iddia edilmez; soket basina log-normal carpanin sigma'si Hecht ve dig.
(2022) grup hedeflerine kalibre edilir; K3'te beklenen bekleme Monte Carlo ile hesaplanir.

Yontem
1) Il x ay TUKETIM payi: EPDK ilk-10 gozlenen; diger iller log(pay)=a+b*log(kW payi) agirligiyla her ayin gercek
   kalanina olceklenir -> il toplamlari EPDK ile birebir.
2) Etkin guc (arac kabul siniri): AC min(P, 11 kW), DC min(P, DC_SINIR).
3) AYRI HAVUZLAR (v3): ilin enerjisi AC ve DC havuzlarina bolunur; ilin DC payi = theta*DC etkin kW /
   (theta*DC etkin kW + AC etkin kW); theta ulusal DC enerji payi EPDK (Tem-Eki 2025) ile tutacak sekilde kalibre.
   Havuz icinde sokete enerji ∝ etkin kW^ALFA. (v2'de istasyon once enerji alip sonra bolundugu icin az AC soketli
   buyuk istasyonlarda AC dolulugu yapay sisiyordu.)
4) Oturum = enerji / EPDK kWh-oturum (AC, DC); uretimde her ayin toplami EPDK adedine olceklenir (kalibrasyon).
5) STOKASTIK HETEROJENLIK: soket carpani M = exp(sigma_g*z - sigma_g^2/2), z~N(0,1), havuz icinde yeniden
   normalize. Gruplar ve hedefler (Hecht ve dig. 2022, PDF metni): L = nominal 4-100 kW soketler -> en yogun %20
   enerjinin %55'i; H = nominal >=100 kW DC -> %40. sigma_L, sigma_H, deterministik yapi + il farklari dahil
   ULUSAL grup icinde hedef tutacak sekilde simulasyonla kalibre edilir. sigma=0'da hedef zaten asiliyorsa 0 kalir.
Dogrulama: ulusal oturum (kalibrasyon oncesi) / EPDK adet raporlanir.
Girdi : ../veri/epdk_ulusal_sarj_*.json, istasyon_duzeltilmis.csv, epdk_panel_il_top10.csv, epdk_panel_aylik.csv, epdk_panel_acdc.csv
Cikti : ../veri/faz2c_soket_lambda.csv (soket x ortalama ay; deterministik beklenen deger + grup sigma)
        ../veri/faz2c_istasyon_lambda_arz.csv (istasyon toplamlari, deterministik)
        ../veri/faz2c_il_ay_pay_arz.csv, ../veri/faz2c_lambda_param.json
Kullanim: python faz2c_arz_lambda.py [--dc250]
"""
import sys, os, glob, json, calendar
import numpy as np, pandas as pd
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
AC_SINIR, DC_SINIR, ALFA = 11.0, (250.0 if "--dc250" in sys.argv else 150.0), 1.0
HEDEF = {"L": 0.55, "H": 0.40}
R_IZGARA, R_SON, TOHUM = 8, 40, 20260916
EK = "_dc250" if "--dc250" in sys.argv else ""

# --- soket tablosu ---
f = sorted(glob.glob(os.path.join(V, "epdk_ulusal_sarj_*.json")))[-1]
d = json.load(open(f, encoding="utf-8"))
rows = d.get("data") or d.get("result") or []
if rows and isinstance(rows[0], list):
    rows = [dict(zip(d["columnNames"], r)) for r in rows]
st = pd.read_csv(os.path.join(V, "istasyon_duzeltilmis.csv"))
st = st[st.hizmet == "HALKA_ACIK"].set_index("ist_no")
sk = []
for r in rows:
    no = r["sarjIstasyonuNo"]
    if no not in st.index:
        continue
    for s in r.get("soketler") or []:
        P = float(s.get("soketGucu") or 0)
        if P <= 0:
            continue
        tip = s.get("soketTipi")
        sk.append((no, st.at[no, "il_ad"], st.at[no, "ilce_ad"], bool(st.at[no, "ada"]), tip, P))
S = pd.DataFrame(sk, columns=["ist_no", "il_ad", "ilce_ad", "ada", "tip", "P_nom"])
S["P_etkin"] = np.where(S.tip == "AC", S.P_nom.clip(upper=AC_SINIR), S.P_nom.clip(upper=DC_SINIR))
S["grup"] = np.select([(S.P_nom >= 4) & (S.P_nom < 100), (S.tip == "DC") & (S.P_nom >= 100)], ["L", "H"], "diger")
print("Soket: %d (AC %d, DC %d) | grup L %d, H %d, diger %d | DC sinir %.0f kW"
      % (len(S), (S.tip == "AC").sum(), (S.tip == "DC").sum(), (S.grup == "L").sum(), (S.grup == "H").sum(), (S.grup == "diger").sum(), DC_SINIR))

# --- il x ay tuketim payi ---
t = pd.read_csv(os.path.join(V, "epdk_panel_il_top10.csv"))
aylar = sorted(t.ay.unique()); M = len(aylar)
kwp = 100 * S.groupby("il_ad").P_nom.sum() / S.P_nom.sum()
x = t.merge(kwp.rename("kwp"), left_on="il", right_index=True)
a, b = np.linalg.lstsq(np.c_[np.ones(len(x)), np.log(x.kwp)], np.log(x.pay), rcond=None)[0]
agr = np.exp(a + b * np.log(kwp))
PAY = pd.DataFrame(index=kwp.index, columns=aylar, dtype=float)
for ay, grp in t.groupby("ay"):
    PAY.loc[grp.il, ay] = grp.pay.values
    dg = PAY.index.difference(grp.il)
    PAY.loc[dg, ay] = (100 - grp.pay.sum()) * agr[dg] / agr[dg].sum()
assert np.allclose(PAY.sum(axis=0), 100)
PAY.round(4).rename_axis("il_ad").to_csv(os.path.join(V, "faz2c_il_ay_pay_arz.csv"), encoding="utf-8-sig")
print("Il duzeyi kW esnekligi: log(pay) = %.3f + %.3f * log(kW payi)" % (a, b))

# --- EPDK ulusal parametreler ---
P_ = pd.read_csv(os.path.join(V, "epdk_panel_aylik.csv")).set_index("ay").reindex(aylar)
C = pd.read_csv(os.path.join(V, "epdk_panel_acdc.csv"))
kwh = {"AC": C.kwh_ac.sum() / C.adet_ac.sum(), "DC": C.kwh_dc.sum() / C.adet_dc.sum()}
dk = {"AC": C.dk_ac.sum() / C.adet_ac.sum(), "DC": C.dk_dc.sum() / C.adet_dc.sum()}
dc_pay_hedef = C.kwh_dc.sum() / (C.kwh_ac.sum() + C.kwh_dc.sum())
dak = np.array([calendar.monthrange(int(m[:4]), int(m[5:]))[1] * 24 * 60 for m in aylar])
E_il = PAY.values / 100 * (P_.mwh.values * 1000)[None, :]                 # il x ay kWh
il_list = list(PAY.index); iix = {n: i for i, n in enumerate(il_list)}
S["il_i"] = S.il_ad.map(iix)
kac = np.bincount(S.il_i[S.tip == "AC"], weights=S.P_etkin[S.tip == "AC"], minlength=len(il_list))
kdc = np.bincount(S.il_i[S.tip == "DC"], weights=S.P_etkin[S.tip == "DC"], minlength=len(il_list))
E_il_ort = E_il.sum(axis=1)


def dc_payi(theta):
    fr = np.where(theta * kdc + kac > 0, theta * kdc / (theta * kdc + kac), 0)
    return (E_il_ort * fr).sum() / E_il_ort.sum() - dc_pay_hedef
theta = brentq(dc_payi, 1e-4, 1e4)
fr_dc = np.where(theta * kdc + kac > 0, theta * kdc / (theta * kdc + kac), 0)

# --- havuzlar: il x tip ---
S["havuz"] = S.il_i * 2 + (S.tip == "DC").astype(int)
nh = len(il_list) * 2
w_det = S.P_etkin.values ** ALFA
havuz_E = np.zeros((nh, M))
havuz_E[0::2] = E_il * (1 - fr_dc)[:, None]
havuz_E[1::2] = E_il * fr_dc[:, None]
hv = S.havuz.values
tipdc = (S.tip == "DC").values
kwh_s = np.where(tipdc, kwh["DC"], kwh["AC"])
dk_s = np.where(tipdc, dk["DC"], dk["AC"])


def soket_enerji(carpan):
    w = w_det * carpan
    pay = w / np.bincount(hv, weights=w, minlength=nh)[hv]
    return pay[:, None] * havuz_E[hv]                                      # soket x ay kWh


E_det = soket_enerji(np.ones(len(S)))
ot = E_det / kwh_s[:, None]
toplam_ot = ot.sum(axis=0)
olcek = P_.adet.values / toplam_ot
print("DOGRULAMA ulusal oturum model/EPDK (kalibrasyon oncesi): ort %.2f, aralik %.2f-%.2f | theta %.3f"
      % ((toplam_ot / P_.adet.values).mean(), (toplam_ot / P_.adet.values).min(), (toplam_ot / P_.adet.values).max(), theta))
AC_SAY = np.bincount(hv[~tipdc], minlength=nh)


def top20(v):
    v = np.sort(v)[::-1]
    return v[: max(1, int(np.ceil(0.2 * len(v))))].sum() / v.sum()


gL, gH = (S.grup == "L").values, (S.grup == "H").values
rng = np.random.default_rng(TOHUM)


def grup_paylari(sL, sH, R):
    sig = np.where(gL, sL, np.where(gH, sH, 0.0))
    out = []
    for _ in range(R):
        z = rng.standard_normal(len(S))
        e = soket_enerji(np.exp(sig * z - sig ** 2 / 2)).mean(axis=1)
        out.append((top20(e[gL]), top20(e[gH])))
    return np.mean(out, axis=0)


p0 = grup_paylari(0.0, 0.0, 1)
print("Deterministik (sigma=0) en yogun %%20 soket enerji payi: L %%%.1f (hedef %%%.0f) | H %%%.1f (hedef %%%.0f)"
      % (100 * p0[0], 100 * HEDEF["L"], 100 * p0[1], 100 * HEDEF["H"]))
izgara = np.round(np.arange(0.0, 2.51, 0.1), 2)
sig = {"L": 0.0, "H": 0.0}
for _ in range(2):                                                          # koordinat bazli kalibrasyon
    for g_, j in (("L", 0), ("H", 1)):
        if (p0[j] >= HEDEF[g_]) and sig[g_] == 0.0:
            continue
        vals = [grup_paylari(s if g_ == "L" else sig["L"], s if g_ == "H" else sig["H"], R_IZGARA)[j] for s in izgara]
        vals = np.maximum.accumulate(vals)
        sig[g_] = float(np.interp(HEDEF[g_], vals, izgara)) if vals[-1] >= HEDEF[g_] else float(izgara[-1])
ulasilan = grup_paylari(sig["L"], sig["H"], R_SON)
print("Kalibre sigma: L %.2f | H %.2f -> ulasilan (R=%d): L %%%.1f | H %%%.1f"
      % (sig["L"], sig["H"], R_SON, 100 * ulasilan[0], 100 * ulasilan[1]))

# --- ciktilar (deterministik beklenen deger; stokastik yapi sigma ile K3'te) ---
ot_k = ot * olcek[None, :]
S["enerji_kwh_ay"] = E_det.mean(axis=1)
S["oturum_ay_kal"] = ot_k.mean(axis=1)
S["rho_det"] = (ot_k * dk_s[:, None] / dak[None, :]).mean(axis=1)
S["sigma"] = np.where(gL, sig["L"], np.where(gH, sig["H"], 0.0))
S.drop(columns=["il_i", "havuz"]).to_csv(os.path.join(V, "faz2c_soket_lambda%s.csv" % EK), index=False, encoding="utf-8-sig")
I = S.groupby("ist_no").agg(il_ad=("il_ad", "first"), ilce_ad=("ilce_ad", "first"), ada=("ada", "first"),
                            n_ac=("tip", lambda x: (x == "AC").sum()), n_dc=("tip", lambda x: (x == "DC").sum()),
                            enerji_kwh_ay=("enerji_kwh_ay", "sum"), oturum_ay_kal=("oturum_ay_kal", "sum"))
I.to_csv(os.path.join(V, "faz2c_istasyon_lambda_arz%s.csv" % EK), encoding="utf-8-sig")
json.dump({"surum": "v3", "AC_SINIR": AC_SINIR, "DC_SINIR": DC_SINIR, "ALFA": ALFA, "theta": theta,
           "il_kw_esneklik": [a, b], "kwh_oturum": kwh, "dk_oturum": dk, "dc_enerji_payi_hedef": dc_pay_hedef,
           "hedef_top20": HEDEF, "sigma": sig, "ulasilan_top20": {"L": ulasilan[0], "H": ulasilan[1]},
           "deterministik_top20": {"L": p0[0], "H": p0[1]}, "aylar": aylar, "olcek_oturum": olcek.tolist()},
          open(os.path.join(V, "faz2c_lambda_param%s.json" % EK), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

m = ~S.ada.values
print("\nSoket ortalama dolulugu (deterministik, ana kara):")
for tp in ("AC", "DC"):
    r = S.loc[m & (S.tip == tp).values, "rho_det"]
    print("  %s: ort %.3f | medyan %.3f | p90 %.3f | p99 %.3f | >0,5: %d" % (tp, r.mean(), r.median(), r.quantile(.9), r.quantile(.99), (r > .5).sum()))
z = rng.standard_normal(len(S))
Es = soket_enerji(np.exp(S.sigma.values * z - S.sigma.values ** 2 / 2))
rs = ((Es / kwh_s[:, None]) * olcek[None, :] * dk_s[:, None] / dak[None, :]).mean(axis=1)
print("Ornek bir stokastik cekim (kalibre sigma):")
for tp in ("AC", "DC"):
    r = rs[m & (S.tip == tp).values]
    print("  %s: ort %.3f | medyan %.3f | p90 %.3f | p99 %.3f | >0,5: %d" % (tp, r.mean(), np.median(r), np.quantile(r, .9), np.quantile(r, .99), (r > .5).sum()))
print("\nDC ortalama dolulugu en yuksek 8 il (deterministik, soket agirlikli):")
print(S[m & tipdc].groupby("il_ad").rho_det.mean().sort_values(ascending=False).head(8).round(3).to_string())
