"""Faz 11 — AS5: iki amacli yatirim optimizasyonu (verimlilik vs Rawlsci esitlik), Pareto egrisi.

KARAR: butce B altinda (i) MEVCUT istasyona soket ekleme [merkezde K3'u giderir] ve
(ii) YENI istasyon acma [cevrede K1-K2'yi giderir] arasinda dagilim.

AMAC 1 (verimlilik): SUM_d P_d * A_d          -> toplam etkin erisim
AMAC 2 (esitlik)   : min_d A_d                 -> Rawlsci; en kotu ilcenin erisimi
epsilon-kisit: AMAC1 maksimize, AMAC2 >= eps; eps taranarak Pareto cikarilir.

DOGRUSALLIK: 2SFCA'da A_d = SUM_j kat[d,j] * S_j^etkin (kat faz10'da hesaplandi, arzdan bagimsiz).
Yalnizca S^etkin, soket sayisinda DOGRUSAL DEGILDIR:
    f_t(n, a) = n * mu_t * (1 - ErlangC(n, a))
Bu fonksiyon a >= 1 icin ICBUKEY DEGILDIR (S-seklindedir: rho<1'e gecerken artan getiri).
Bu yuzden kesen/zarf yaklasimi GECERSIZDIR; her soket sayisi icin AYRI IKILI degiskenle
TAM temsil kullanilir (soket sayisi tamsayi oldugundan bu yaklasiklik degil, kesin temsildir).

MALIYET (kurumsal teknik raporlar: ICCT 2019 calisma raporu 2019-14, INL/RPT-22-68598; iki butce duzeyi):
  AC portu 8.000 USD (aralik 3-15 bin) · DC 150 kW portu 100.000 USD (aralik 80-120 bin)
  yeni istasyon sabit saha/sebeke maliyeti 50.000 USD
  Dayanak: donanim toplam maliyetin yalnizca %20-35'i; kalani elektrik/saha isi -> sabit kalem ayri.

VARSAYIMLAR (makalede acikca yazilacak):
  - Talep (lambda) disardan verilidir; yeni arz talebi degistirmez (ikame/rekabet modellenmez).
  - Yeni istasyonun soket basina yuku, senaryodaki o turun MEDYAN doluluguna esit alinir.
  - 2SFCA paydasi arzdan bagimsiz oldugu icin yeni istasyon mevcutlarin paydasini degistirmez.

Girdi : ../veri/faz10_kat_mevcut.npz, faz10_kat_aday.npz, faz10_indeks.json, faz6_istasyon_k3*.csv
Cikti : ../veri/faz11_pareto*.csv, faz11_cozum*.csv
Kullanim: python faz11_optimizasyon.py [--pi 1] [--butce 100e6] [--nokta 12]
"""
import sys, os, json, time
import numpy as np, pandas as pd
import gurobipy as gp
from gurobipy import GRB

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
PI = sys.argv[sys.argv.index("--pi") + 1] if "--pi" in sys.argv else "1"
BUTCE = float(sys.argv[sys.argv.index("--butce") + 1]) if "--butce" in sys.argv else 100e6
NOKTA = int(sys.argv[sys.argv.index("--nokta") + 1]) if "--nokta" in sys.argv else 12
KADD = KNEW = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 6   # soket ust siniri
_arg = lambda ad, vars: type(vars)(sys.argv[sys.argv.index(ad) + 1]) if ad in sys.argv else vars
FIYAT = {"AC": _arg("--acfiyat", 8_000.0), "DC": _arg("--dcfiyat", 100_000.0)}   # revizyon: duyarlilik
SABIT = _arg("--sabit", 50_000.0)
NEXP = _arg("--nexp", 1200)                                                  # revizyon: aday kumesi
GAP = _arg("--gap", 0.005)                                                   # revizyon:
TLIM = _arg("--sure", 300.0)
NOKTA_SEC = [int(x) for x in sys.argv[sys.argv.index("--noktalar") + 1].split(",")] if "--noktalar" in sys.argv else None
SON = "" if PI == "1" else "_pi%s" % PI
t0 = time.time()


def erlang_c(c, a):
    c = int(c)
    if c <= 0 or a <= 0:
        return 0.0
    if a / c >= 1:
        return 1.0
    k = np.arange(c + 1)
    lg = k * np.log(a) - np.cumsum(np.r_[0.0, np.log(np.arange(1, c + 1))])
    mx = lg.max()
    pay = np.exp(lg[-1] - np.log(1 - a / c) - mx)
    return float(pay / (pay + np.exp(lg[:-1] - mx).sum()))


IX = json.load(open(os.path.join(V, "faz10_indeks.json"), encoding="utf-8"))
sid = IX["sid"]; nd = len(sid)
MV = np.load(os.path.join(V, "faz10_kat_mevcut.npz"))
AD = np.load(os.path.join(V, "faz10_kat_aday.npz"))
P_d = MV["P_d"]; n0 = {"AC": MV["n_ac"], "DC": MV["n_dc"]}
print("Ilce %d | mevcut katsayi %d | aday katsayi %d | %.0f sn" % (nd, len(MV["v"]), len(AD["v"]), time.time() - t0), flush=True)

par = json.load(open(os.path.join(V, "faz2c_lambda_param.json"), encoding="utf-8"))
MU = {t: 60.0 / par["dk_oturum"][t] for t in ("AC", "DC")}
k3 = pd.read_csv(os.path.join(V, "faz6_istasyon_k3%s.csv" % SON))
ist_no = IX["ist_no"]
A_yuk = {t: pd.Series(k3[k3.tip == t].set_index("ist_no").a).reindex(ist_no).fillna(0.0).values for t in ("AC", "DC")}
RHO_MED = {t: float(k3[k3.tip == t].rho.median()) for t in ("AC", "DC")}
print("mu: AC %.4f, DC %.4f | yeni istasyon varsayilan doluluk: AC %.4f, DC %.4f"
      % (MU["AC"], MU["DC"], RHO_MED["AC"], RHO_MED["DC"]), flush=True)

# --- katsayilari tur bazinda sozluklestir ---
def ayikla(Z):
    out = {}
    for ti, t in ((0, "AC"), (1, "DC")):
        m = Z["t"] == ti
        out[t] = (Z["d"][m], Z["j"][m], Z["v"][m])
    return out


KM_, KA_ = ayikla(MV), ayikla(AD)

# --- taban (hicbir yatirim yok) ---
A0 = np.zeros(nd)
for t in ("AC", "DC"):
    d_, j_, v_ = KM_[t]
    S0 = np.array([n0[t][j] * MU[t] * (1 - erlang_c(n0[t][j], A_yuk[t][j])) for j in range(len(ist_no))])
    np.add.at(A0, d_, v_ * S0[j_])
print("TABAN: toplam erisim %.4g | min ilce %.6g | sifir erisimli ilce %d (%.2fM kisi)"
      % (float((P_d * A0).sum()), float(A0.min()), int((A0 == 0).sum()), float(P_d[A0 == 0].sum()) / 1e6), flush=True)

# --- genisleme adayi istasyonlar: nufus agirlikli katki kutlesi + en kotu ilceye katki ---
sec = {}
for t in ("AC", "DC"):
    d_, j_, v_ = KM_[t]
    nj = len(ist_no)
    kutle = np.bincount(j_, weights=P_d[d_] * v_, minlength=nj)
    enbuyuk = np.zeros(nj); np.maximum.at(enbuyuk, j_, v_)
    a1 = set(np.argsort(-kutle)[:NEXP].tolist()); a2 = set(np.argsort(-enbuyuk)[:NEXP].tolist())
    sec[t] = sorted((a1 | a2) & set(np.unique(j_).tolist()))
    print("[%s] genisleme adayi istasyon: %d" % (t, len(sec[t])), flush=True)

# --- f degerleri ---
def f_mevcut(t, j, k):
    n = n0[t][j] + k
    return n * MU[t] * (1 - erlang_c(n, A_yuk[t][j]))


def f_yeni(t, k):
    if k <= 0:
        return 0.0
    return k * MU[t] * (1 - erlang_c(k, k * RHO_MED[t]))


na = len(IX["aday_sid"])
M = gp.Model("AS5")
M.Params.OutputFlag = 0
M.Params.MIPGap = GAP
M.Params.TimeLimit = TLIM

y = {}      # mevcut istasyon genislemesi
for t in ("AC", "DC"):
    for j in sec[t]:
        y[t, j] = M.addVars(KADD + 1, vtype=GRB.BINARY, name="y_%s_%d" % (t, j))
        M.addConstr(y[t, j].sum() == 1)
u = {}      # aday istasyon soketleri
for t in ("AC", "DC"):
    for c in range(na):
        u[t, c] = M.addVars(KNEW + 1, vtype=GRB.BINARY, name="u_%s_%d" % (t, c))
        M.addConstr(u[t, c].sum() == 1)
z = M.addVars(na, vtype=GRB.BINARY, name="z")
for c in range(na):
    for t in ("AC", "DC"):
        M.addConstr(z[c] >= 1 - u[t, c][0])

# --- ilce erisimi ---
Aexp = [gp.LinExpr(A0[d]) for d in range(nd)]
for t in ("AC", "DC"):
    d_, j_, v_ = KM_[t]
    sj = set(sec[t])
    for idx in range(len(v_)):
        j = int(j_[idx])
        if j not in sj:
            continue
        d, v = int(d_[idx]), float(v_[idx])
        taban = f_mevcut(t, j, 0)
        Aexp[d] += v * gp.quicksum((f_mevcut(t, j, k) - taban) * y[t, j][k] for k in range(1, KADD + 1))
    d_, c_, v_ = KA_[t]
    for idx in range(len(v_)):
        d, c, v = int(d_[idx]), int(c_[idx]), float(v_[idx])
        Aexp[d] += v * gp.quicksum(f_yeni(t, k) * u[t, c][k] for k in range(1, KNEW + 1))

# --- maliyet ---
maliyet = gp.quicksum(FIYAT[t] * k * y[t, j][k] for t in ("AC", "DC") for j in sec[t] for k in range(1, KADD + 1)) \
    + gp.quicksum(FIYAT[t] * k * u[t, c][k] for t in ("AC", "DC") for c in range(na) for k in range(1, KNEW + 1)) \
    + gp.quicksum(SABIT * z[c] for c in range(na))
M.addConstr(maliyet <= BUTCE)

verim = gp.quicksum(float(P_d[d]) * Aexp[d] for d in range(nd))
mmin = M.addVar(lb=0.0, name="min_ilce")
for d in range(nd):
    M.addConstr(mmin <= Aexp[d])
M.update()
print("Model: %d degisken (%d ikili), %d kisit | %.0f sn" % (M.NumVars, M.NumBinVars, M.NumConstrs, time.time() - t0), flush=True)

# --- revizyon (02.10.2026): yalnizca BASLANGICTA sifir olan ilceler icin esik ---
# Pareto adimi esigi BUTUN ilcelere uygular (mmin); burada esik yalnizca A0 = 0 olan ilcelere konur ve
# verim maksimize edilir. Boylece "dokuz ilceyi erisime kavusturmanin" en dusuk verim kaybi bulunur.
if "--hedef9" in sys.argv:
    sifir = np.nonzero(A0 <= 1e-12)[0]
    theta10 = float(np.quantile(A0[A0 > 1e-12], 0.10))
    esikler = [("pareto_ilk_adim_tabani", float(sys.argv[sys.argv.index("--hedef9") + 1])),
               ("pozitif_ilce_10_yuzdelik", theta10)]
    M.setObjective(verim, GRB.MAXIMIZE); M.optimize()
    V_max = verim.getValue()
    print("Sifir ilce %d | verim max %.5f" % (len(sifir), V_max), flush=True)
    sat9 = []
    for ad, th in esikler:
        kis = [M.addConstr(Aexp[d] >= th) for d in sifir]
        M.optimize()
        Ad = np.array([Aexp[d].getValue() for d in range(nd)])
        har_yeni = sum(FIYAT[t] * k * u[t, c][k].X for t in ("AC", "DC") for c in range(na) for k in range(1, KNEW + 1)) \
            + sum(SABIT * z[c].X for c in range(na))
        acik = [any(u[t, c][k].X > 0.5 for t in ("AC", "DC") for k in range(1, KNEW + 1)) for c in range(na)]
        r9 = {"esik": ad, "theta": th, "verim": verim.getValue(), "verim_max": V_max,
              "kayip_%": 100 * (V_max - verim.getValue()) / V_max, "yeni_istasyon": int(sum(acik)),
              "yeni_istasyon_butce": har_yeni, "dokuz_ilce_min": float(Ad[sifir].min()),
              "ulusal_min": float(Ad.min()), "sifir_ilce": int((Ad <= 1e-12).sum()),
              "gap_%": 100 * M.MIPGap, "sure_sn": M.Runtime}
        sat9.append(r9)
        print("  %-26s theta %.3e | verim %.2f | kayip %%%.3f | yeni ist %d (%.2f M$) | 9 ilce min %.3e | ulusal min %.3e | gap %%%.3f"
              % (ad, th, r9["verim"], r9["kayip_%"], r9["yeni_istasyon"], har_yeni / 1e6, r9["dokuz_ilce_min"],
                 r9["ulusal_min"], r9["gap_%"]), flush=True)
        M.remove(kis)
    pd.DataFrame(sat9).to_csv(os.path.join(V, "faz11c_dokuz_ilce_hedefi.csv"), index=False, encoding="utf-8-sig")
    print("Kaydedildi: faz11c_dokuz_ilce_hedefi.csv")
    sys.exit(0)

# --- uc noktalar ---
M.setObjective(verim, GRB.MAXIMIZE); M.optimize()
V_max, E_at_Vmax = verim.getValue(), mmin.X
M.setObjective(mmin, GRB.MAXIMIZE); M.optimize()
E_max = mmin.X
# NOT: mmin'i TEK BASINA maksimize eden cozum, verimlilik acisindan optimaller arasinda keyfidir;
# bu yuzden "esitlik ucundaki verim" degeri epsilon-kisit taramasindan okunur, buradan DEGIL.
print("Uc noktalar: verim max %.5g (o cozumde esitlik %.4g) | ulasilabilir en yuksek esitlik %.4g"
      % (V_max, E_at_Vmax, E_max), flush=True)

ilce = pd.read_csv(os.path.join(V, "faz0_ilce.csv")).set_index("shapeID")
sege = ilce.sege_skor.reindex(sid).values
ust = float(np.nanquantile(sege, 2 / 3))
aday_sid = IX["aday_sid"]
aday_sege = ilce.sege_skor.reindex(aday_sid).values

esm = pd.read_csv(os.path.join(V, "istasyon_ilce_eslesme.csv"))
kim = "ist_no" if "ist_no" in esm.columns else "no"      # dosyada kimlik sutunu 'no'
mp = esm.drop_duplicates(kim).set_index(kim).shapeID
ist_sege = ilce.sege_skor.reindex(mp.reindex(ist_no).values).values
print("Istasyon-SEGE eslesmesi: %d/%d (kimlik sutunu '%s') | merkez esigi SEGE >= %.3f"
      % (int(np.isfinite(ist_sege).sum()), len(ist_no), kim, ust), flush=True)

sat, karar = [], []
M.setObjective(verim, GRB.MAXIMIZE)
for i, eps in enumerate(np.linspace(E_at_Vmax, E_max, NOKTA)):
    if NOKTA_SEC and (i + 1) not in NOKTA_SEC:
        continue
    kis = M.addConstr(mmin >= eps)
    M.optimize()
    if M.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or M.SolCount == 0:
        M.remove(kis); continue
    har_mev = sum(FIYAT[t] * k * y[t, j][k].X for t in ("AC", "DC") for j in sec[t] for k in range(1, KADD + 1))
    har_yeni = sum(FIYAT[t] * k * u[t, c][k].X for t in ("AC", "DC") for c in range(na) for k in range(1, KNEW + 1)) \
        + sum(SABIT * z[c].X for c in range(na))
    mrk = sum(FIYAT[t] * k * y[t, j][k].X for t in ("AC", "DC") for j in sec[t] for k in range(1, KADD + 1)
              if np.isfinite(ist_sege[j]) and ist_sege[j] >= ust)
    # yeni istasyonun merkez payina soket maliyeti YANINDA sabit saha maliyeti de eklenir (payda ile tutarli)
    yeni_mrk = sum(SABIT * z[c].X for c in range(na) if np.isfinite(aday_sege[c]) and aday_sege[c] >= ust) \
        + sum((FIYAT[t] * k * u[t, c][k].X for t in ("AC", "DC") for c in range(na) for k in range(1, KNEW + 1)
                    if np.isfinite(aday_sege[c]) and aday_sege[c] >= ust))
    Ad = np.array([Aexp[d].getValue() for d in range(nd)])
    sifir_n = int((Ad <= 1e-12).sum()); sifir_p = float(P_d[Ad <= 1e-12].sum())
    # soket kurulan aday = acik istasyon (sabit maliyet 0 iken z serbestce 1 olabilir; sayima z degil soket esas)
    acik = [any(u[t, c][k].X > 0.5 for t in ("AC", "DC") for k in range(1, KNEW + 1)) for c in range(na)]
    yeni_sege = [aday_sege[c] for c in range(na) if acik[c] and np.isfinite(aday_sege[c])]
    sat.append({"eps": eps, "verim": verim.getValue(), "esitlik": mmin.X,
                "sifir_erisim_ilce": sifir_n, "sifir_erisim_nufus": sifir_p,
                "yeni_ist_medyan_SEGE": float(np.median(yeni_sege)) if yeni_sege else np.nan,
                "butce_mevcut_soket": har_mev, "butce_yeni_istasyon": har_yeni,
                "yeni_istasyon_sayisi": int(sum(acik)),
                "merkez_pay_%": 100 * (mrk + yeni_mrk) / max(har_mev + har_yeni, 1),
                "gap_%": 100 * M.MIPGap, "amac": M.ObjVal, "en_iyi_sinir": M.ObjBound, "sure_sn": M.Runtime,
                "yeni_istasyon_butce_payi_%": 100 * har_yeni / BUTCE})
    print("  eps %2d/%d: verim %.5g | esitlik %.3e | mevcut %.1fM/yeni %.1fM USD | yeni ist %d (medyan SEGE %.2f) | merkez %%%.0f | sifir ilce %d (%.2fM kisi) | gap %%%.2f"
          % (i + 1, NOKTA, verim.getValue(), mmin.X, har_mev / 1e6, har_yeni / 1e6,
             sat[-1]["yeni_istasyon_sayisi"], sat[-1]["yeni_ist_medyan_SEGE"],
             sat[-1]["merkez_pay_%"], sifir_n, sifir_p / 1e6, 100 * M.MIPGap), flush=True)
    for t in ("AC", "DC"):
        for j in sec[t]:
            k = next(k for k in range(KADD + 1) if y[t, j][k].X > 0.5)
            if k:
                karar.append({"nokta": i + 1, "eps": eps, "tur": "mevcut", "soket": t, "kimlik": ist_no[j],
                              "k": k, "sege": ist_sege[j]})
        for c in range(na):
            k = next(k for k in range(KNEW + 1) if u[t, c][k].X > 0.5)
            if k:
                karar.append({"nokta": i + 1, "eps": eps, "tur": "yeni", "soket": t, "kimlik": aday_sid[c],
                              "k": k, "sege": aday_sege[c]})
    for d in range(nd):
        karar.append({"nokta": i + 1, "eps": eps, "tur": "ilce_erisim", "soket": "", "kimlik": sid[d],
                      "k": float(Ad[d]), "sege": sege[d]})
    M.remove(kis)

R = pd.DataFrame(sat)
CIKTI_AD = "faz11_pareto%s_B%dM%s%s.csv" % (SON, int(round(BUTCE / 1e6)), "" if KADD == 6 else "_K%d" % KADD,
                                           sys.argv[sys.argv.index("--ek") + 1] if "--ek" in sys.argv else "")   # butce dosya adinda: kosumlar birbirinin ustune yazmasin
R.to_csv(os.path.join(V, CIKTI_AD), index=False, encoding="utf-8-sig")
pd.DataFrame(karar).to_csv(os.path.join(V, CIKTI_AD.replace("pareto", "cozum")), index=False, encoding="utf-8-sig")
print("\nKaydedildi: %s | toplam %.0f sn" % (CIKTI_AD, time.time() - t0))
