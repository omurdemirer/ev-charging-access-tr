"""Faz 11b — Sifir ortalama erisimli ilceleri kapsamanin ASGARI maliyeti revizyon.

Ana model (faz11) tam butceli bir cozumun yatirim bilesimini verir; oradaki 6,2 milyon dolar asgari
kapsama maliyeti degildir. Burada yalnizca YENI istasyonlarla, her sifir-erisimli ilcenin erisimini
bir esigin (theta) ustune cikaran en ucuz yatirim cozulur:
    min  sum_c [ SABIT z_c + sum_t sum_k FIYAT_t k u_ctk ]
    s.t. sum_{c,t,k} kat_dct g_t(k) u_ctk >= theta     (her sifir-erisimli ilce d)
         sum_k u_ctk <= 1 ;  z_c >= u_ctk
g_t(k) = k mu_t (1 - ErlangC(k, k rho_med,t)), faz11 ile ayni. Mevcut istasyonlar bu ilcelere hicbir
havzada ulasmadigi icin (kat = 0) modelde yer almaz.
Esikler: (i) kume kapsama (her ilceye en az bir yeni soketin havzasi ulasir; 1e-9 gibi bir esik
Gurobi olurluluk toleransinin altinda kaldigi icin kullanilmaz), (ii) theta = sifirdan buyuk ilce erisimlerinin 10. yuzdeligi.
Her cozum icin secilen istasyonlardan km agi uzerinde havza icinde kalan koken nufusu hesaplanir:
dokuz ilcenin kendi nufusunun ne kadari gercekten kapsaniyor (ilce ortalamasi > 0 bunu garanti etmez).
Cikti: ../veri/faz11b_asgari_kapsama.csv
"""
import json, os, sys
import numpy as np, pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ag_ortak import Ag

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
FIYAT, SABIT, KNEW, MIL = {"AC": 8000.0, "DC": 100000.0}, 50000.0, 6, 1.609344
HAVZA = {"AC": 5 * MIL, "DC": 25 * MIL}


def erlang_c_1(c, a):
    """faz11 ile ayni skaler Erlang-C (rho >= 1 ise 1)."""
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


def erlang_c(c, a):
    return np.array([erlang_c_1(ci, ai) for ci, ai in zip(np.atleast_1d(c), np.atleast_1d(a))])

IX = json.load(open(os.path.join(V, "faz10_indeks.json"), encoding="utf-8"))
MV, AD = np.load(os.path.join(V, "faz10_kat_mevcut.npz")), np.load(os.path.join(V, "faz10_kat_aday.npz"))
par = json.load(open(os.path.join(V, "faz2c_lambda_param.json"), encoding="utf-8"))
MU = {t: 60.0 / par["dk_oturum"][t] for t in ("AC", "DC")}
k3 = pd.read_csv(os.path.join(V, "faz6_istasyon_k3.csv"))
RHO_MED = {t: float(k3[k3.tip == t].rho.median()) for t in ("AC", "DC")}
g = {t: [0.0] + [k * MU[t] * (1 - erlang_c(np.array([k]), np.array([k * RHO_MED[t]]))[0]) for k in range(1, KNEW + 1)]
     for t in ("AC", "DC")}

# taban ilce erisimi (yatirimsiz)
sid = IX["sid"]; nd = len(sid); ist_no = IX["ist_no"]
A_yuk = {t: pd.Series(k3[k3.tip == t].set_index("ist_no").a).reindex(ist_no).fillna(0.0).values for t in ("AC", "DC")}
n0 = {"AC": MV["n_ac"], "DC": MV["n_dc"]}
A0 = np.zeros(nd)
for ti, t in ((0, "AC"), (1, "DC")):
    m = MV["t"] == ti
    S0 = n0[t] * MU[t] * (1 - erlang_c(n0[t].astype(int), A_yuk[t]))
    np.add.at(A0, MV["d"][m], MV["v"][m] * S0[MV["j"][m]])
sifir = np.nonzero(A0 <= 0)[0]
theta10 = float(np.quantile(A0[A0 > 0], 0.10))
print("Sifir ortalama erisimli ilce: %d | pozitif ilce erisimi 10. yuzdelik = %.3e" % (len(sifir), theta10))

aday = IX["aday_sid"]; na = len(aday)
kat = {}
for ti, t in ((0, "AC"), (1, "DC")):
    m = AD["t"] == ti
    for d, c, v in zip(AD["d"][m], AD["j"][m], AD["v"][m]):
        if d in set(sifir):
            kat[(int(d), int(c), t)] = float(v)
C_ilgili = sorted({c for (_, c, _) in kat})
print("Bu ilcelere ulasan aday konum: %d" % len(C_ilgili))

M_ilce = pd.read_csv(os.path.join(V, "osm", "ilce_merkez.csv")).set_index("shapeID")
K = pd.read_csv(os.path.join(V, "osm", "izgara_koken.csv"))
ag = Ag(os.path.join(V, "osm"))
Gk_r = ag.Gk.T.tocsr()
KOORD = {c: (float(M_ilce.loc[aday[c]].lon), float(M_ilce.loc[aday[c]].lat)) for c in C_ilgili}

# revizyon: aday konum kisiti testi. --aday_ek N ile her sifir-erisimli ilcede, ilce merkezine
# ek olarak en kalabalik N koken hucresi aday konum olur. Katsayilar faz10_katsayi.py ile AYNI tanimla
# (kenara baglama, iki uc dugumden ters graf aramasi, tum Turkiye kokenleri uzerinden 2SFCA paydasi) hesaplanir.
NEK = int(sys.argv[sys.argv.index("--aday_ek") + 1]) if "--aday_ek" in sys.argv else 0
if NEK:
    def W_gauss(dd, p):
        return np.where(dd <= p, (np.exp(-dd * dd / (2 * p * p)) - np.exp(-0.5)) / (1 - np.exp(-0.5)), 0.0)
    sira = np.argsort(K.dugum.values, kind="stable")
    o_dug = K.dugum.values[sira].astype(np.int64); o_pop = K.nufus.values[sira]
    o_bag = K.baglanma_km.values[sira]; o_sid = K.shapeID.values[sira]
    sid_i = {s_: i for i, s_ in enumerate(sid)}
    o_di = np.array([sid_i.get(s_, -1) for s_ in o_sid], dtype=np.int64)
    ptr = np.searchsorted(o_dug, np.arange(ag.N + 1))
    P_d = MV["P_d"]
    ek = (K[K.shapeID.isin([sid[d] for d in sifir])].sort_values("nufus", ascending=False)
          .groupby("shapeID").head(NEK).reset_index(drop=True))
    hs = ag.kenara_bagla(ek.lon.values, ek.lat.values)
    for i in range(len(ek)):
        c = na + i; KOORD[c] = (float(ek.lon[i]), float(ek.lat[i]))
        ga, gb = int(hs["ga"][i]), int(hs["gb"][i])
        Dk = dijkstra(Gk_r, directed=True, indices=[ga, gb], limit=max(HAVZA.values()) + 0.1)
        ulas = np.nonzero(np.isfinite(Dk[0]) | np.isfinite(Dk[1]))[0]
        say = ptr[ulas + 1] - ptr[ulas]; ulas = ulas[say > 0]; say = say[say > 0]
        oi = np.repeat(ptr[ulas], say) + (np.arange(say.sum()) - np.repeat(np.cumsum(say) - say, say))
        dn = np.repeat(np.arange(len(ulas)), say)
        fr = hs["frac"][i]
        ka = Dk[0][ulas] + fr * hs["e_km"][i] if hs["fwd"][i] else np.full(len(ulas), np.inf)
        kb = Dk[1][ulas] + (1 - fr) * hs["e_km"][i] if hs["rev"][i] else np.full(len(ulas), np.inf)
        d_km = np.minimum(ka, kb)[dn] + o_bag[oi] + hs["baglanma_km"][i]
        for t in ("AC", "DC"):
            W = W_gauss(d_km, HAVZA[t]); nz = (W > 0) & (o_di[oi] >= 0)
            if not nz.any():
                continue
            ii, w = oi[nz], W[nz]; talep = float(np.sum(o_pop[ii] * w))
            pay = np.bincount(o_di[ii], weights=o_pop[ii] * w, minlength=nd)
            for d in sifir:
                if pay[d] > 0:
                    kat[(int(d), c, t)] = float(pay[d] / max(P_d[d], 1e-9) / talep)
    C_ilgili = sorted({c for (_, c, _) in kat})
    print("Ek aday: ilce basina %d koken hucresi, toplam %d | bu ilcelere ulasan aday konum: %d"
          % (NEK, len(ek), len(C_ilgili)), flush=True)


def kapsa_vektor(c, t, hedef):
    """(c, t) adayinin hedef kokenlerden hangilerini havza icinde kapsadigi (kapsama() ile ayni ag erisimi)."""
    hd = hedef.dugum.values.astype(int)
    lo, la = KOORD[c]
    hs = ag.kenara_bagla(np.array([lo]), np.array([la]))
    ga, gb = int(hs["ga"][0]), int(hs["gb"][0])
    Dk = dijkstra(Gk_r, directed=True, indices=[ga, gb], limit=HAVZA[t] + 0.1)
    fr = hs["frac"][0]
    ka = Dk[0][hd] + fr * hs["e_km"][0] if hs["fwd"][0] else np.full(len(hd), np.inf)
    kb = Dk[1][hd] + (1 - fr) * hs["e_km"][0] if hs["rev"][0] else np.full(len(hd), np.inf)
    return (np.minimum(ka, kb) + hedef.baglanma_km.values + hs["baglanma_km"][0]) < HAVZA[t]


def kapsama(secim):
    """Secilen (aday c, tur t) cozumunun dokuz ilcedeki koken nufusu kapsama orani.
    revizyon: faz10_katsayi.py ile AYNI ag erisimi kullanilir: ilce merkezi en yakin
    KENARA baglanir (kenar ici konum + yon izinleri + baglanti bacagi), ters grafta iki uc dugumden arama
    yapilir ve Gauss agirligi pozitif (d < havza) olan kokenler kapsanmis sayilir."""
    hedef = K[K.shapeID.isin([sid[d] for d in sifir])]
    hd = hedef.dugum.values.astype(int)
    kaps = np.zeros(len(hedef), bool)
    for c, t in secim:
        lo, la = KOORD[c]
        hs = ag.kenara_bagla(np.array([lo]), np.array([la]))
        ga, gb = int(hs["ga"][0]), int(hs["gb"][0])
        Dk = dijkstra(Gk_r, directed=True, indices=[ga, gb], limit=HAVZA[t] + 0.1)
        fr = hs["frac"][0]
        ka = Dk[0][hd] + fr * hs["e_km"][0] if hs["fwd"][0] else np.full(len(hd), np.inf)
        kb = Dk[1][hd] + (1 - fr) * hs["e_km"][0] if hs["rev"][0] else np.full(len(hd), np.inf)
        dd = np.minimum(ka, kb) + hedef.baglanma_km.values + hs["baglanma_km"][0]
        kaps |= dd < HAVZA[t]
    return float(hedef.nufus[kaps].sum()), float(hedef.nufus.sum())


sat = []
for ad, theta in (("kume_kapsama", None), ("yuzdelik10", theta10)):
    M = gp.Model(); M.Params.OutputFlag = 0; M.Params.MIPGap = 0.0; M.Params.Threads = 4
    u = {(c, t, k): M.addVar(vtype=GRB.BINARY) for c in C_ilgili for t in ("AC", "DC") for k in range(1, KNEW + 1)}
    z = {c: M.addVar(vtype=GRB.BINARY) for c in C_ilgili}
    for c in C_ilgili:
        for t in ("AC", "DC"):
            M.addConstr(gp.quicksum(u[c, t, k] for k in range(1, KNEW + 1)) <= 1)
            for k in range(1, KNEW + 1):
                M.addConstr(z[c] >= u[c, t, k])
    for d in sifir:
        if theta is None:   # kume kapsama: ilceye ulasan en az bir yeni soket (sayisal toleranstan bagimsiz)
            M.addConstr(gp.quicksum(u[c, t, k] for (dd, c, t) in kat if dd == d for k in range(1, KNEW + 1)) >= 1)
        else:
            M.addConstr(gp.quicksum(kat[(int(d), c, t)] * g[t][k] * u[c, t, k] for (dd, c, t) in kat if dd == d
                                    for k in range(1, KNEW + 1)) >= theta)
    M.setObjective(gp.quicksum(SABIT * z[c] for c in C_ilgili)
                   + gp.quicksum(FIYAT[t] * k * u[c, t, k] for (c, t, k) in u), GRB.MINIMIZE)
    M.optimize()
    if "--kapsama_max" in sys.argv:
        # revizyon: ikinci asama. Asgari maliyet sabitken dokuz ilcede kapsanan koken nufusu en buyuklenir
        # (birinci asamada esit maliyetli cozumler arasindaki secim kapsamaya kayitsizdir).
        opt = M.ObjVal; hedef = K[K.shapeID.isin([sid[d] for d in sifir])].reset_index(drop=True)
        kap_set = {}
        for c in C_ilgili:
            for t in ("AC", "DC"):
                kap_set[c, t] = np.nonzero(kapsa_vektor(c, t, hedef))[0]
        x = M.addVars(len(hedef), lb=0.0, ub=1.0, name="x")
        ort = {}
        for (c, t), ii in kap_set.items():
            for o in ii:
                ort.setdefault(int(o), []).append((c, t))
        for o in range(len(hedef)):
            M.addConstr(x[o] <= gp.quicksum(u[c, t, k] for (c, t) in ort.get(o, []) for k in range(1, KNEW + 1)))
        maliyet = (gp.quicksum(SABIT * z[c] for c in C_ilgili)
                   + gp.quicksum(FIYAT[t] * k * u[c, t, k] for (c, t, k) in u))
        kapsanan = gp.quicksum(float(hedef.nufus[o]) * x[o] for o in range(len(hedef)))
        if "--kapsama_hedef" in sys.argv:   # dokuz ilce nufusunun en az q'sunu kapsayan asgari maliyet
            q = float(sys.argv[sys.argv.index("--kapsama_hedef") + 1])
            M.addConstr(kapsanan >= q * float(hedef.nufus.sum()))
            M.setObjective(maliyet, GRB.MINIMIZE); M.Params.MIPGap = 1e-4
            M.optimize(); opt = M.ObjVal
        else:
            M.addConstr(maliyet <= opt + 1.0)
            M.setObjective(kapsanan, GRB.MAXIMIZE)
            M.optimize()
    secim = [(c, t) for (c, t, k) in u if u[c, t, k].X > 0.5]
    kap, top = kapsama(secim)
    r = {"esik": ad, "theta": theta, "aday_ek": NEK, "ek_secilen": int(sum(z[c].X > 0.5 for c in C_ilgili if c >= na)),
         "maliyet_usd": opt if "--kapsama_max" in sys.argv else M.ObjVal, "yeni_istasyon": int(sum(z[c].X > 0.5 for c in C_ilgili)),
         "ac_soket": int(sum(k for (c, t, k) in u if t == "AC" and u[c, t, k].X > 0.5)),
         "dc_soket": int(sum(k for (c, t, k) in u if t == "DC" and u[c, t, k].X > 0.5)),
         "kapsanan_nufus": kap, "dokuz_ilce_nufus": top, "kapsama_%": 100 * kap / top, "gap": M.MIPGap}
    sat.append(r)
    print("%-15s maliyet %.3f M$ | yeni istasyon %d (AC soket %d, DC soket %d) | dokuz ilce nufusunun %%%.1f'i kapsanir"
          % (ad, r["maliyet_usd"] / 1e6, r["yeni_istasyon"], r["ac_soket"], r["dc_soket"], r["kapsama_%"]))
CK = "faz11b_asgari_kapsama%s%s.csv" % ("_ek%d" % NEK if NEK else "", ("_kmax" + ("_q" + sys.argv[sys.argv.index("--kapsama_hedef") + 1] if "--kapsama_hedef" in sys.argv else "")) if "--kapsama_max" in sys.argv else "")
pd.DataFrame(sat).to_csv(os.path.join(V, CK), index=False, encoding="utf-8-sig")
print("Kaydedildi:", CK)
