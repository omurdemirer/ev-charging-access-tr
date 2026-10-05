"""Faz 4c (v2) — Yonlu kenar batarya enerjisi (kWh). K2 (topografya) kanalinin cekirdegi.

AYRISTIRMA (v2, 18.09.2026 — v1'deki negatif-kenar sorununu kokten cozer):
  E(u->v) = surtunme/eta_dt + m g Dpos/eta_dt - m g Dneg*eta_re*eta_dt + P_aux t
  surtunme = sum( m g C_rr d + 0,5 rho C_d A v^2 d )   [>=0]
  net = z(v) - z(u)  (DUGUM rakimi; graf genelinde TEK ve tutarli)
  C   = kenar ici dalgalanma fazlasi = (Dpos_ham + Dneg_ham - |net|)/2  [>=0]
  Dpos = max(net,0) + C ,  Dneg = max(-net,0) + C     ->  Dpos - Dneg = net  (TANIM GEREGI)

POTANSIYEL DONUSUMU: phi(n) = m g z(n) * eta_re * eta_dt   (KUCUK olan INIS katsayisi)
  w' = E + phi(u) - phi(v) = surtunme/eta_dt + m g Dpos (1/eta_dt - eta_re*eta_dt) >= 0
  Negatiflik MATEMATIKSEL OLARAK imkansiz; en kisa yollar degismez (Johnson potansiyeli).
  v1'de potansiyel CEKIS verimiyle (1/eta_dt) kurulmustu; yokus yukari kenarlarda asiri
  duserek %8,76 negatif birakiyordu. Inis katsayisi dogru secimdir.
  KAYNAK DOGRULANACAK: Eisner, Funke & Storandt (AAAI 2011).

DEM gurultusu: 150 m pencereli duzlestirme sonrasi bile kisa segmentlerde artik gurultu var
(segmentlerin %16,8'i >%10 egim gosteriyor — gercek yol agi icin imkansiz). Bu yuzden segment
egimi |%20| ile kirpilir; kirpmanin tasidigi toplam |dz| payi %1,9'dur. Net rakim farki ise
kirpmadan DEGIL, tutarli dugum rakimindan alinir; boylece kirpma potansiyeli bozmaz.

Girdi : ../veri/osm/{ag.npz, ag_geometri.npz, ag_yukselti.npz}, ../veri/faz4_enerji_param.json
Cikti : ../veri/osm/ag_enerji.npz
"""
import os, sys, json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OSM = os.path.join(HERE, "..", "veri", "osm")
P = json.load(open(os.path.join(HERE, "..", "veri", "faz4_enerji_param.json"), encoding="utf-8"))
m = P["arac"]["m_kg"]["deger"]
Y = P["arac"]["yuvarlanma"]; Cr, c1, c2 = Y["Cr"], Y["c1"], Y["c2"]
Cd, A = P["arac"]["C_d"]["deger"], P["arac"]["A_m2"]["deger"]
edt = P["arac"]["eta_dt"]["deger"]
ere = P["arac"]["eta_regen"]["deger"]
if "--regen" in sys.argv:                      # duyarlilik: eta_regen taramasi
    ere = float(sys.argv[sys.argv.index("--regen") + 1])
PauxkW = P["arac"]["P_aux_kW"]["deger"]
if "--paux" in sys.argv:                       # AS4: mevsimsel yardimci yuk (Fiori: 35C 1,2 / 5C 2,2 kW)
    PauxkW = float(sys.argv[sys.argv.index("--paux") + 1])
EK = ("_regen%g" % ere if "--regen" in sys.argv else "") + ("_paux%g" % PauxkW if "--paux" in sys.argv else "") + ("_duz" if "--duz" in sys.argv else "")
rho, gg = P["cevre"]["rho_kgm3"]["deger"], P["cevre"]["g_ms2"]["deger"]
EGS = P["sayisal"]["egim_sinir"]["deger"]
if "--egim" in sys.argv:                       # 7a duyarlilik: egim siniri (varsayilan %20)
    EGS = float(sys.argv[sys.argv.index("--egim") + 1])
    EK += "_egim%g" % EGS
V2 = "--v2" in sys.argv                         # eski bilesen bazli formulasyon (yalniz karsilastirma icin)
if V2:
    EK += "_v2"
J2K = 3.6e6

zd = np.load(os.path.join(OSM, "ag_yukselti.npz"))["z_duz"].astype(np.float64)
if "--duz" in sys.argv:                        # M2 kontrolu: duz arazi (yukselti etkisi sifir; yalniz hiz + yardimci yuk)
    zd = np.zeros_like(zd)
G = np.load(os.path.join(OSM, "ag_geometri.npz"))
a = np.load(os.path.join(OSM, "ag.npz"))
U, V, km, dk = a["U"], a["V"], a["km"], a["dk"]
pa, pb, ters = a["pos_a"], a["pos_b"], a["ters"].astype(bool)
E, R = len(U), 6371000.0

la_, lo_ = np.radians(G["lat"]), np.radians(G["lon"])
d_g = np.hypot((lo_[1:] - lo_[:-1]) * np.cos(0.5 * (la_[1:] + la_[:-1])), la_[1:] - la_[:-1]) * R
dz_g = np.diff(zd)
print("Kenar %d | kose %d | m=%g kg, Cd=%g, A=%g, eta_dt=%.4f, eta_regen=%.3f, P_aux=%.2f kW" % (E, len(zd), m, Cd, A, edt, ere, PauxkW), flush=True)

L = (pb - pa).astype(np.int64); top = int(L.sum())
eid = np.repeat(np.arange(E, dtype=np.int64), L)
idx = np.arange(top, dtype=np.int64) - np.repeat(np.cumsum(np.r_[0, L])[:-1], L) + np.repeat(pa.astype(np.int64), L)
d = d_g[idx]
dz = np.where(ters[eid], -dz_g[idx], dz_g[idx])
dz = np.clip(dz, -EGS * d, EGS * d)
v = np.repeat((km / np.maximum(dk, 1e-9)) * 1000.0 / 60.0, L)
# Fiori: yuvarlanma direnci HIZA BAGLI -> F = m g (Cr/1000)(c1*v_kmh + c2). Sabit C_rr degil.
yuv = (Cr / 1000.0) * (c1 * (v * 3.6) + c2)
fr_s = m * gg * yuv * d + 0.5 * rho * Cd * A * v * v * d          # parca surtunme + hava direnci (J)
fric = np.bincount(eid, weights=fr_s, minlength=E)
del yuv
# v3 (30.09.2026, revizyon:): Denk. (6) PARCA bazinda uygulanir. Parca mekanik enerjisi
# Em = fr + m g dz; cekis ise Em/eta_dt, geri kazanim ise Em*eta_re*eta_dt.
kre = m * gg * ere * edt                                          # potansiyel katsayisi (J/m)
Em = fr_s + m * gg * dz
c_s = np.where(Em > 0, Em / edt, Em * ere * edt)
sum_c = np.bincount(eid, weights=c_s, minlength=E)
sum_dz = np.bincount(eid, weights=dz, minlength=E)
# indirgenmis parca maliyeti c_s - kre*dz >= 0 (kanit: Em>0 ise fr/edt + m g dz(1/edt - ere*edt) veya
# (fr + m g dz)/edt - kre dz; Em<=0 ise fr*ere*edt). Denetim:
red = c_s - kre * dz
print("Indirgenmis parca maliyeti: en dusuk %.3e J (negatif parca %d)" % (red.min(), int((red < -1e-6).sum())), flush=True)
del Em, c_s, red, fr_s
Dpos_h = np.bincount(eid, weights=np.maximum(dz, 0.0), minlength=E)
Dneg_h = np.bincount(eid, weights=np.maximum(-dz, 0.0), minlength=E)
del d, dz, v, idx, eid

# --- TEK ve TUTARLI dugum rakimi (ayni dugume gelen tum kenarlarin ortalamasi) ---
bas, son = np.where(ters, pb, pa), np.where(ters, pa, pb)
n = len(a["n_lon"]); tp = np.zeros(n); ad = np.zeros(n)
for nd, pos in ((U, bas), (V, son)):
    np.add.at(tp, nd, zd[pos]); np.add.at(ad, nd, 1.0)
zn = np.where(ad > 0, tp / np.maximum(ad, 1), 0.0)

net = zn[V] - zn[U]
C = np.maximum((Dpos_h + Dneg_h - np.abs(net)) / 2.0, 0.0)     # kenar ici dalgalanma fazlasi
Dpos, Dneg = np.maximum(net, 0.0) + C, np.maximum(-net, 0.0) + C
if V2:
    kwh = (fric / edt + m * gg * Dpos / edt - m * gg * Dneg * ere * edt) / J2K + PauxkW * (dk / 60.0)
else:
    # parca toplamı + dugum yukseltisiyle uzlastirma: kenarin net yukselti farki DUGUM rakimindan alinir,
    # boylece potansiyel donusumu tutarli kalir: w = sum(c_s - kre dz_s) + P_aux t >= 0
    kwh = (sum_c + kre * (net - sum_dz)) / J2K + PauxkW * (dk / 60.0)
    # revizyon (02.10.2026): uzlastirma teriminin buyuklugu nicel olarak raporlanir
    duz = kre * (net - sum_dz) / J2K
    fark_m = np.abs(net - sum_dz)
    print("Uzlastirma terimi: |net - sum dz| medyan %.2f m, 95. yuzdelik %.2f m | |terim| ort %.5f kWh "
          "(ort |kenar enerjisi| %.4f kWh, oran %%%.2f) | toplam |terim| / toplam |kenar| %%%.2f"
          % (np.median(fark_m), np.quantile(fark_m, 0.95), np.abs(duz).mean(), np.abs(kwh).mean(),
             100 * np.abs(duz).mean() / np.abs(kwh).mean(), 100 * np.abs(duz).sum() / np.abs(kwh).sum()), flush=True)

phi = m * gg * zn * ere * edt / J2K
kwh_d = kwh + phi[U] - phi[V]
neg = int((kwh_d < -1e-12).sum())
print("Donusum sonrasi negatif kenar: %d (%%%.6f) | en dusuk %.3e kWh" % (neg, 100 * neg / E, kwh_d.min()), flush=True)
kwh_d = np.maximum(kwh_d, 0.0)

np.savez_compressed(os.path.join(OSM, "ag_enerji%s.npz" % EK), kwh=kwh, kwh_donusmus=kwh_d, phi=phi,
                    z_dugum=zn, tirmanis_m=Dpos, dusus_m=Dneg, net_m=net)
ok = km > 0.001
oran = kwh[ok] / km[ok]
print("\n--- OZET ---")
ebar = float(kwh[ok].sum() / km[ok].sum())
json.dump({"ebar_kWh_per_km": ebar, "eta_regen": ere},
          open(os.path.join(HERE, "..", "veri", "faz4_ebar%s.json" % EK), "w", encoding="utf-8"), indent=2)
print("Mesafe agirlikli tuketim : %.2f kWh/100 km  (ebar = %.4f kWh/km -> faz4_ebar.json)" % (100 * ebar, ebar))
print("Medyan kenar tuketimi    : %.2f kWh/100 km" % (100 * np.median(oran)))
print("Yuzdelik (kWh/100km)     : p5 %.1f | p25 %.1f | p75 %.1f | p95 %.1f"
      % tuple(100 * np.percentile(oran, [5, 25, 75, 95])))
print("Net geri kazanimli kenar : %%%.1f" % (100 * (kwh < 0).mean()))
print("Dugum rakimi             : %.0f..%.0f m (ort %.0f)" % (zn.min(), zn.max(), zn.mean()))
# --- yon asimetrisi: ayni (u,v) cifti ters yonuyle ---
anah = U.astype(np.int64) * n + V
ters_anah = V.astype(np.int64) * n + U
sira = np.argsort(anah); poz = np.searchsorted(anah[sira], ters_anah)
poz = np.clip(poz, 0, len(sira) - 1); es = sira[poz]
gec = anah[es] == ters_anah
print("Cift yonlu kenar : %d | ortalama |E(u,v)-E(v,u)| = %.4f kWh (ort kenar %.4f kWh)"
      % (gec.sum(), np.abs(kwh[gec] - kwh[es[gec]]).mean(), kwh.mean()))
