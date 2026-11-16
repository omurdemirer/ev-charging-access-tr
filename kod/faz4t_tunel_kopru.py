"""Faz 4t — Tunel ve kopru yollarinin OSM kimlikleri (revizyon).

Yukselti modeli yer yuzeyini verir; tunelde arazinin, koprude vadinin yuksekligi yol yuksekligi sanilir.
Bu betik ag kurulumundaki yol siniflarindan tunnel/bridge etiketli olanlari ayiklar. faz4_enerji_kenar.py
--tunel secenegi bu yollarin yukseltisini uc noktalar (portal/ayak) arasinda dogrusal ara degerle degistirir.
Girdi : ../veri/osm/turkey-latest.osm.pbf (ag ile ayni anlik goruntu), ../veri/osm/ag.npz (w_osm)
Cikti : ../veri/osm/tunel_kopru.npz  (osm_id, tur: 1 tunel, 2 kopru)
"""
import os, sys
import numpy as np
import osmium

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faz1_ag_kur import ascii_calisma_kopyasi, SINIF  # noqa: E402

OSM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri", "osm")
_, pbf = ascii_calisma_kopyasi()
ids, tur = [], []
for o in osmium.FileProcessor(pbf).with_filter(osmium.filter.KeyFilter("highway")):
    if not o.is_way() or o.tags.get("highway") not in SINIF:
        continue
    t, b = o.tags.get("tunnel", "no"), o.tags.get("bridge", "no")
    if t not in ("no", "") :
        ids.append(o.id); tur.append(1)
    elif b not in ("no", ""):
        ids.append(o.id); tur.append(2)
ids, tur = np.array(ids, dtype=np.int64), np.array(tur, dtype=np.int8)
w_osm = np.load(os.path.join(OSM, "ag.npz"))["w_osm"]
agda = np.isin(ids, w_osm)
np.savez_compressed(os.path.join(OSM, "tunel_kopru.npz"), osm_id=ids, tur=tur)
print("Tunel yolu %d, kopru yolu %d | agdaki yollarla eslesen %d / %d" % ((tur == 1).sum(), (tur == 2).sum(), agda.sum(), len(ids)))
