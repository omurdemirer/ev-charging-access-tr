"""Faz 4b — Ag geometrisinin her koseginde yukselti ornekle + kenar bazinda egim profili cikar.
Girdi : ../veri/osm/ag_geometri.npz (8,38 M kose), ../veri/dem/*.tif (Copernicus GLO-90)
Cikti : ../veri/osm/ag_yukselti.npz  (kose yukseltisi z, metre; ve way bazinda kumulatif tirmanis)

Yontem notu (makalede raporlanacak):
  Ham DEM dusey hatasi GLO-90 icin ~2-4 m; 90 m yatay adimda bu +-%3-4 sahte egim uretir.
  Bu yuzden yukselti profili, yol boyunca MESAFE PENCERELI hareketli ortalama ile duzlestirilir
  (pencere ZBOY metre). Egim, duzlestirilmis profilden hesaplanir. Duzlestirme yapilmazsa
  yuvarlama gurultusu hem tirmanisi hem rejeneratif geri kazanimi sistematik olarak sisirir.
"""
import os, sys, glob
import numpy as np, rasterio

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OSM = os.path.join(HERE, "..", "veri", "osm")
DEM = os.path.join(HERE, "..", "veri", "dem")
ZBOY = 150.0   # duzlestirme penceresi (m)

g = np.load(os.path.join(OSM, "ag_geometri.npz"))
lon, lat, wid = g["lon"], g["lat"], g["wid"]
N = len(lon)
z = np.full(N, np.nan, dtype=np.float32)
print("Kose %d | pafta dosyasi %d" % (N, len(glob.glob(os.path.join(DEM, "*.tif")))), flush=True)

anahtar = (np.floor(lat).astype(np.int32) + 90) * 1000 + (np.floor(lon).astype(np.int32) + 180)
sira = np.argsort(anahtar, kind="stable")
ak = anahtar[sira]
sinir = np.r_[0, np.flatnonzero(np.diff(ak)) + 1, len(ak)]
bulunan = eksik = 0
for b in range(len(sinir) - 1):
    idx = sira[sinir[b]:sinir[b + 1]]
    la = int(ak[sinir[b]] // 1000) - 90; lo = int(ak[sinir[b]] % 1000) - 180
    ad = "Copernicus_DSM_COG_30_%s%02d_00_%s%03d_00_DEM.tif" % ("N" if la >= 0 else "S", abs(la), "E" if lo >= 0 else "W", abs(lo))
    yol = os.path.join(DEM, ad)
    if not os.path.exists(yol):
        z[idx] = 0.0; eksik += len(idx); continue     # pafta yok = deniz seviyesi
    with rasterio.open(yol) as src:
        A = src.read(1)
        inv = ~src.transform
        c, r = inv * (lon[idx], lat[idx])
        r = np.clip(np.round(r).astype(np.int32), 0, A.shape[0] - 1)
        c = np.clip(np.round(c).astype(np.int32), 0, A.shape[1] - 1)
        v = A[r, c].astype(np.float32)
        v[v < -1000] = 0.0                            # nodata
        z[idx] = v; bulunan += len(idx)
    if (b + 1) % 20 == 0:
        print("  pafta %d/%d" % (b + 1, len(sinir) - 1), flush=True)
print("Orneklendi: %d | paftasiz (deniz) %d | z %.0f..%.0f m, ort %.0f" % (bulunan, eksik, np.nanmin(z), np.nanmax(z), np.nanmean(z)), flush=True)

# --- way bazinda mesafe pencereli duzlestirme ---
R = 6371000.0
wsinir = np.r_[0, np.flatnonzero(np.diff(wid)) + 1, N]
zd = z.copy()
for k in range(len(wsinir) - 1):
    a, b = wsinir[k], wsinir[k + 1]
    if b - a < 3: continue
    la_, lo_ = np.radians(lat[a:b]), np.radians(lon[a:b])
    dx = np.r_[0.0, np.cumsum(np.hypot((lo_[1:] - lo_[:-1]) * np.cos(0.5 * (la_[1:] + la_[:-1])), la_[1:] - la_[:-1]) * R)]
    zi = z[a:b].astype(np.float64)
    cs = np.r_[0.0, np.cumsum(zi)]
    sol = np.searchsorted(dx, dx - ZBOY / 2, "left")
    sag = np.searchsorted(dx, dx + ZBOY / 2, "right")
    zd[a:b] = ((cs[sag] - cs[sol]) / np.maximum(sag - sol, 1)).astype(np.float32)
    if (k + 1) % 50000 == 0: print("  way %d/%d" % (k + 1, len(wsinir) - 1), flush=True)

np.savez_compressed(os.path.join(OSM, "ag_yukselti.npz"), z=z, z_duz=zd, zboy=np.float32(ZBOY))
print("\nKaydedildi: ag_yukselti.npz | ham-duz fark std %.2f m" % np.nanstd(z - zd))
