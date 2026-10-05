"""Faz 4a — Copernicus GLO-90 DEM paftalarini indir (acik erisim, kimlik dogrulamasi yok).
Kaynak: Copernicus DEM GLO-90, ESA/Airbus, AWS Open Data (s3://copernicus-dem-90m, anonim HTTPS).
Lisans: Copernicus DEM free, full and open (ESA 2021). Atif: European Space Agency (2021),
        Copernicus Global Digital Elevation Model, https://doi.org/10.5270/ESA-c5d3d65
Yalnizca ag geometrisinin dustugu 1x1 derece paftalar indirilir.
"""
import os, sys, collections, urllib.request, urllib.error
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
DEM = os.path.join(HERE, "..", "veri", "dem")
os.makedirs(DEM, exist_ok=True)
BASE = "https://copernicus-dem-90m.s3.amazonaws.com"

g = np.load(os.path.join(HERE, "..", "veri", "osm", "ag_geometri.npz"))
hucre = sorted(collections.Counter(zip(np.floor(g["lat"]).astype(int), np.floor(g["lon"]).astype(int))).keys())
print("Gerekli pafta: %d" % len(hucre), flush=True)

ok = atlanan = hata = 0
for i, (la, lo) in enumerate(hucre):
    ad = "Copernicus_DSM_COG_30_%s%02d_00_%s%03d_00_DEM" % ("N" if la >= 0 else "S", abs(la), "E" if lo >= 0 else "W", abs(lo))
    hedef = os.path.join(DEM, ad + ".tif")
    if os.path.exists(hedef) and os.path.getsize(hedef) > 1024:
        atlanan += 1; continue
    try:
        urllib.request.urlretrieve("%s/%s/%s.tif" % (BASE, ad, ad), hedef)
        ok += 1
    except urllib.error.HTTPError as e:
        # 404 = tamamen deniz olan pafta; DEM yok, normal
        if os.path.exists(hedef): os.remove(hedef)
        hata += 1
        print("  yok (%s): %s" % (e.code, ad), flush=True)
    if (i + 1) % 20 == 0:
        print("  %d/%d | indi %d, mevcut %d, yok %d" % (i + 1, len(hucre), ok, atlanan, hata), flush=True)

boyut = sum(os.path.getsize(os.path.join(DEM, f)) for f in os.listdir(DEM) if f.endswith(".tif"))
print("\nBitti: indi %d | zaten vardi %d | pafta yok (deniz) %d | toplam %.2f GB" % (ok, atlanan, hata, boyut / 1e9))
