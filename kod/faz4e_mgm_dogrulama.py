"""Faz 4e — Yukselti dogrulamasi: MGM resmi il merkezi olcum noktasi rakimlari (revizyon, 29.09.2026).

Eski faz4d'nin 20 referans rakami kaynaksizdi (genel bilgi); bu betik onun yerine gecer.
Referans: MGM il tahmin sayfalarindaki Rakim / Enlem / Boylam (veri/mgm_il_merkez_rakim.json, 81 il).
Iki karsilastirma:
  (a) ham Copernicus GLO-90 DEM, MGM noktasinin kendi koordinatinda (veri kaynaginin dogrulugu)
  (b) modelin kullandigi duzlestirilmis yol agi yukseltisi, noktaya en yakin ag kosesinde
Kural: MGM'de rakimi tam 0 raporlanan kayitlar (eksik deger olabilir) ayrica gosterilir, ozet
istatistik hem dahil hem haric verilir.
Cikti: ../veri/faz4e_mgm_dogrulama.csv + ekrana ozet
"""
import glob, json, math, os, sys
import numpy as np, pandas as pd
import rasterio
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
M = json.load(open(os.path.join(V, "mgm_il_merkez_rakim.json"), encoding="utf-8"))["veri"]
f = lambda s: float(s.replace(",", "."))
R = pd.DataFrame([{"il": k, "mgm_rakim": f(v[0]), "lat": f(v[1]), "lon": f(v[2])} for k, v in M.items()])
# il sayfasinda 0 raporlanan 4 kayit: ayni olcum noktasi icin resmi kaynaktan tamamlama (MGM ilce sayfasi / WMO OSCAR)
T = json.load(open(os.path.join(V, "mgm_il_merkez_rakim.json"), encoding="utf-8"))["tamamlama"]
R["rakim_kaynagi"] = "MGM il sayfasi"
for il, v in T.items():
    if il.startswith("_"):
        continue
    i = R.index[R.il == il][0]
    R.loc[i, ["mgm_rakim", "rakim_kaynagi"]] = [float(v["rakim"]), v["kaynak"]]

# (a) ham DEM
def pafta(lat, lon):
    return os.path.join(V, "dem", "Copernicus_DSM_COG_30_N%02d_00_E%03d_00_DEM.tif" % (math.floor(lat), math.floor(lon)))

dem = []
for la, lo in zip(R.lat, R.lon):
    p = pafta(la, lo)
    if not os.path.exists(p):
        dem.append(np.nan); continue
    with rasterio.open(p) as r:
        dem.append(float(next(r.sample([(lo, la)]))[0]))
R["dem_ham"] = dem

# (b) ag yukseltisi (en yakin kose)
G = np.load(os.path.join(V, "osm", "ag_geometri.npz"))
zd = np.load(os.path.join(V, "osm", "ag_yukselti.npz"))["z_duz"]
kx = np.cos(np.radians(39.0))
t = cKDTree(np.c_[G["lon"] * kx, G["lat"]])
dist, idx = t.query(np.c_[R.lon * kx, R.lat])
R["ag_z"] = zd[idx].astype(float)
R["ag_mesafe_km"] = dist * 111.32
R["sifir_rakim"] = R.mgm_rakim == 0

def ozet(x, y, etiket):
    d = (y - x).dropna(); xx = x[d.index]
    r = np.corrcoef(xx, y[d.index])[0, 1]
    print("%-34s n=%2d | ort. mutlak sapma %5.1f m | medyan mutlak %5.1f m | ort. sapma %+6.1f m | r %.4f"
          % (etiket, len(d), d.abs().mean(), d.abs().median(), d.mean(), r))
    return {"etiket": etiket, "n": len(d), "ort_mutlak": d.abs().mean(), "medyan_mutlak": d.abs().median(),
            "ort_sapma": d.mean(), "r": r}

print("Ag kosesine uzaklik: medyan %.2f km, en buyuk %.2f km" % (R.ag_mesafe_km.median(), R.ag_mesafe_km.max()))
oz = []
for ad, g in (("tum 81 il", R), ("rakimi 0 olanlar haric", R[~R.sifir_rakim])):
    oz.append(ozet(g.mgm_rakim, g.dem_ham, "(a) ham DEM, %s" % ad))
    oz.append(ozet(g.mgm_rakim, g.ag_z, "(b) ag yukseltisi, %s" % ad))
print("\nRakimi 0 raporlanan kayitlar:")
print(R[R.sifir_rakim][["il", "mgm_rakim", "dem_ham", "ag_z"]].round(1).to_string(index=False))
d = (R.dem_ham - R.mgm_rakim).abs()
print("\nEn buyuk 5 sapma (ham DEM, 0 haric):")
print(R[~R.sifir_rakim].assign(fark=d).nlargest(5, "fark")[["il", "mgm_rakim", "dem_ham", "ag_z", "fark"]].round(1).to_string(index=False))
R.to_csv(os.path.join(V, "faz4e_mgm_dogrulama.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(oz).to_csv(os.path.join(V, "faz4e_mgm_ozet.csv"), index=False, encoding="utf-8-sig")
print("\nKaydedildi: faz4e_mgm_dogrulama.csv, faz4e_mgm_ozet.csv")
