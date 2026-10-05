"""Faz 6 — K3 (TIKANIKLIK) kanali: M/M/c kuyruk ile ETKIN ARZ.

SECIM (18.09.2026): "etkin hizmet orani" yaklasimi — SERBEST PARAMETRE YOK.
  Etkin arz  S_j^K3 = S_j^K1 * (1 - C(c_j, a_j))
  C = Erlang-C: bir gelisin BEKLEMEK ZORUNDA kalma olasiligi.  (1 - C) = ANINDA hizmet olasiligi.
Neden bu: bekleme cezali S/(1+alfa*Wq) bicimi alfa gibi kalibre edilmemis bir serbest parametre
gerektirir ve dogrulanamayan bir varsayim ekler. Erlang-C'nin kendisi yalnizca (c, a)'ya baglidir;
yuk arttikca (1-C) KONVEKS bicimde duser — tasarim v3'un "konveks bekleme" sarti saglanir.

M/M/c kurulumu (AC ve DC AYRI havuzlar; faz2c v3 karari):
  c_j   = istasyondaki o turden soket sayisi
  mu    = 60 / dk_oturum  (saatlik hizmet orani; AC 82,95 dk -> 0,7233/sa, DC 36,41 dk -> 1,6478/sa)
  lambda_j = istasyonun o turden saatlik oturum gelis orani (faz2c oturum_ay_kal'dan)
  a = lambda/mu (Erlang), rho = a/c

YUK SENARYOSU (pi): oturum_ay_kal AYLIK toplamdir; saatlik ortalamaya bolunur (30,44*24 saat).
  pi = 1,0  BIRINCIL (ortalama yuk)
  pi = 1,5  ZIRVE senaryosu. Dayanak: Hecht, Figgener & Sauer (2022) gun ici EN YUKSEK/EN DUSUK
            doluluk orani 1,4-1,8 (kaynagin tam metninden dogrulandi). Zirve/ORTALAMA katsayisi kaynakta
            YOKTUR; 1,5 bu araliktan turetilmis bir SENARYODUR, olculmus bir sabit degildir.
  ⚠️ Daha onceki notlarda gecen "DC haftalik zirve 2,09-2,22" rakaminin DOGRULANMIS KAYNAGI YOK;
     kullanilmamistir.

Girdi : ../veri/faz2c_soket_lambda.csv, ../veri/faz2c_lambda_param.json
Cikti : ../veri/faz6_istasyon_k3.csv (istasyon x tip: c, lambda, a, rho, ErlangC, S_K1, S_K3)
Kullanim: python faz6_kuyruk_k3.py [--pi 1.5]
"""
import sys, os, json
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
SAAT_AY = 30.44 * 24.0
PI = float(sys.argv[sys.argv.index("--pi") + 1]) if "--pi" in sys.argv else 1.0
SEN = sys.argv[sys.argv.index("--senaryo") + 1] if "--senaryo" in sys.argv else None
MC = int(sys.argv[sys.argv.index("--mc") + 1]) if "--mc" in sys.argv else 0   # revizyon: Monte Carlo tekrar
TOHUM = 20260930
T_BEKLE = 10.0 / 60.0                                  # revizyon: 10 dk icinde hizmet (saat)


def erlang_c(c, a):
    """P(bekleme) — log-uzayinda, tasma olmadan. c: tamsayi dizi, a: Erlang yuku dizi."""
    c = np.asarray(c, dtype=np.int64); a = np.asarray(a, dtype=np.float64)
    C = np.zeros(len(a))
    gec = (c > 0) & (a > 0)
    rho = np.where(gec & (c > 0), a / np.maximum(c, 1), 0.0)
    kararli = gec & (rho < 1.0)
    for cc in np.unique(c[gec]):
        m = gec & (c == cc)
        k = np.arange(cc + 1)
        # log terimler: a^k/k!
        lg = k[None, :] * np.log(a[m])[:, None] - np.cumsum(np.r_[0.0, np.log(np.arange(1, cc + 1))])[None, :]
        son = lg[:, -1]                                   # a^c/c!
        rhom = np.minimum(a[m] / cc, 1 - 1e-12)
        # Erlang-C = [a^c/c! * 1/(1-rho)] / [sum_{k<c} a^k/k! + a^c/c! * 1/(1-rho)]
        mx = np.maximum(son - np.log(1 - rhom), lg[:, :-1].max(axis=1))
        pay = np.exp(son - np.log(1 - rhom) - mx)
        payda = pay + np.exp(lg[:, :-1] - mx[:, None]).sum(axis=1)
        C[m] = pay / payda
    C[gec & ~kararli] = 1.0                               # rho >= 1: her gelis bekler
    return C


par = json.load(open(os.path.join(V, "faz2c_lambda_param.json"), encoding="utf-8"))
sk = pd.read_csv(os.path.join(V, "faz2c_soket_lambda.csv"))
g = sk.groupby(["ist_no", "tip"]).agg(c=("P_etkin", "size"), oturum_ay=("oturum_ay_kal", "sum"),
                                      kw=("P_etkin", "sum"), il_ad=("il_ad", "first"),
                                      ilce_ad=("ilce_ad", "first")).reset_index()
g["mu_sa"] = g.tip.map({t: 60.0 / par["dk_oturum"][t] for t in ("AC", "DC")})
g["lambda_sa"] = g.oturum_ay / SAAT_AY * PI
if SEN:   # AS4: il bazinda mevsimsel mekansal indeks x ulusal seviye
    _s = json.load(open(os.path.join(V, "faz8_senaryo.json"), encoding="utf-8"))[SEN]
    _c = g.il_ad.map(_s["il_indeks"]).fillna(1.0).values * _s["seviye"]
    g["lambda_sa"] = g.lambda_sa * _c
    g["senaryo"] = SEN
    print("SENARYO %s: seviye %.4f | il carpani ort %.4f (min %.3f, max %.3f)"
          % (SEN, _s["seviye"], _c.mean(), _c.min(), _c.max()))
g["a"] = g.lambda_sa / g.mu_sa
g["rho"] = g.a / g.c
g["ErlangC"] = erlang_c(g.c.values, g.a.values)
g["aninda_hizmet"] = 1.0 - g.ErlangC
g["S_K1"] = g.c * g.mu_sa                      # saatlik oturum kapasitesi (faz3'teki K1)
g["S_K3"] = g.S_K1 * g.aninda_hizmet
g["S_K1g"] = g.kw
g["S_K3g"] = g.kw * g.aninda_hizmet
g["pi"] = PI
# alternatif gosterge (revizyon): P(W <= t) = 1 - C exp(-(c mu - lambda) t); kararsiz sistemde 0
_kar = g.rho < 1.0
g["hizmet10"] = np.where(_kar, 1.0 - g.ErlangC * np.exp(-(g.c * g.mu_sa - g.lambda_sa).clip(lower=0) * T_BEKLE), 0.0)
g.loc[g.lambda_sa <= 0, "hizmet10"] = 1.0

# Monte Carlo (revizyon): faz2c'de kalibre edilen log-normal soket carpani UYGULANIR.
# Soket carpani M = exp(sigma z - sigma^2/2), z~N(0,1); il x tur havuzu icinde yeniden normalize
# edilir (il talep toplamlari korunur). Istasyon lambdasi soket toplamidir; Erlang-C yeniden hesaplanir.
if MC:
    _hav = sk.il_ad.astype(str) + "|" + sk.tip
    _oran = g.lambda_sa.values / np.maximum(g.oturum_ay.values, 1e-12)          # PI ve senaryo carpani
    _anahtar = pd.MultiIndex.from_frame(sk[["ist_no", "tip"]])
    _gi = pd.MultiIndex.from_frame(g[["ist_no", "tip"]]).get_indexer(_anahtar)
    for r in range(MC):
        z = np.random.default_rng(TOHUM + r).standard_normal(len(sk))
        M = np.exp(sk.sigma.values * z - sk.sigma.values ** 2 / 2)
        o = sk.oturum_ay_kal.values * M
        top0 = pd.Series(sk.oturum_ay_kal.values).groupby(_hav.values).transform("sum").values
        top1 = pd.Series(o).groupby(_hav.values).transform("sum").values
        o = o * top0 / np.maximum(top1, 1e-12)
        ist = np.bincount(_gi, weights=o, minlength=len(g))
        a_r = ist * _oran / g.mu_sa.values
        g["aninda_mc%d" % r] = 1.0 - erlang_c(g.c.values, a_r)
    print("Monte Carlo: %d tekrar (tohum %d...) | aninda hizmet ort. (deterministik %.4f) -> MC ort. %.4f"
          % (MC, TOHUM, g.aninda_hizmet.mean(), g[["aninda_mc%d" % r for r in range(MC)]].values.mean()))

ek = ("_" + SEN) if SEN else ("" if PI == 1.0 else "_pi%g" % PI)
g.to_csv(os.path.join(V, "faz6_istasyon_k3%s.csv" % ek), index=False, encoding="utf-8-sig")

print("YUK SENARYOSU pi = %.2f" % PI)
print("Istasyon-tip satiri %d | AC %d, DC %d" % (len(g), (g.tip == "AC").sum(), (g.tip == "DC").sum()))
for t in ("AC", "DC"):
    q = g[g.tip == t]
    w = q.c
    print("\n--- %s ---  mu = %.4f oturum/sa (oturum %.1f dk)" % (t, q.mu_sa.iloc[0], par["dk_oturum"][t]))
    print("  c: medyan %d, ort %.2f | lambda: medyan %.4f/sa | a: medyan %.4f" %
          (int(q.c.median()), q.c.mean(), q.lambda_sa.median(), q.a.median()))
    print("  rho (doluluk)   : medyan %.4f | p90 %.4f | p99 %.4f | rho>=1 olan istasyon: %d" %
          (q.rho.median(), q.rho.quantile(.9), q.rho.quantile(.99), (q.rho >= 1).sum()))
    print("  Erlang-C        : medyan %.4f | p90 %.4f | soket-agirlikli ort %.4f" %
          (q.ErlangC.median(), q.ErlangC.quantile(.9), np.average(q.ErlangC, weights=w)))
    print("  etkin arz kaybi : soket-agirlikli %%%.2f  (S_K3 / S_K1 = %.4f)" %
          (100 * (1 - q.S_K3.sum() / q.S_K1.sum()), q.S_K3.sum() / q.S_K1.sum()))
print("\nKaydedildi: faz6_istasyon_k3%s.csv" % ek)
