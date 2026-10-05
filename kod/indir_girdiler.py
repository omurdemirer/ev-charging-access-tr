"""Download the public input data used in the study into ../veri (run once, before the pipeline).

Automatic downloads (public sources, see README for licences and access dates):
  EPDK monthly charging-market reports (PDF, June 2024 - July 2026) -> ../veri/epdk_aylik/epdk_sarj_YYYY-MM.pdf
  OpenStreetMap extract for Turkiye (Geofabrik)                     -> ../veri/osm/turkey-latest.osm.pbf
  WorldPop R2025A constrained 1 km population, 2025                  -> ../veri/worldpop/
  geoBoundaries gbOpen TUR ADM1 / ADM2                               -> ../veri/
  SEGE-2022 district socio-economic development ranking (PDF)       -> ../veri/SEGE-2022_ilce.pdf
  KGM Traffic and transportation information 2025 (PDF)             -> ../veri/kgm/
Other inputs are fetched by their own scripts (epdk_ulusal_cek.py, faz4_dem_indir.py) or are manual
(TUIK tables); see README.

NOTE: Geofabrik only serves the *latest* extract. The study used the snapshot of 2026-09-15
(md5 b48de84f41355cb1348d1ec3bb003a80). The processed road network built from that snapshot is
archived on Zenodo (ODbL 1.0) so that the results can be reproduced exactly.
Usage: python indir_girdiler.py
"""
import json, os, re, sys, urllib.request

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
AY = {"Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4, "Mayıs": 5, "Haziran": 6, "Temmuz": 7, "Ağustos": 8,
      "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12}
DOSYALAR = [
    ("https://download.geofabrik.de/europe/turkey-latest.osm.pbf", "osm/turkey-latest.osm.pbf"),
    ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/TUR/v1/1km_ua/constrained/"
     "tur_pop_2025_CN_1km_R2025A_UA_v1.tif", "worldpop/tur_pop_2025_CN_1km_R2025A_UA_v1.tif"),
    ("https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/TUR/ADM1/geoBoundaries-TUR-ADM1.geojson",
     "geoBoundaries-TUR-ADM1.geojson"),
    ("https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/TUR/ADM2/geoBoundaries-TUR-ADM2.geojson",
     "geoBoundaries-TUR-ADM2.geojson"),
    ("https://baka.gov.tr/assets/upload/dosyalar/2022-ilce-sege_opt_1.pdf", "SEGE-2022_ilce.pdf"),
    ("https://www.kgm.gov.tr/SiteCollectionDocuments/KGMdocuments/Istatistikler/TrafikveUlasimBilgileri/"
     "25TrafikUlasimBilgileri.pdf", "kgm/kgm_2025_trafik_ulasim_bilgileri.pdf"),
]


def indir(url, hedef):
    yol = os.path.join(V, hedef)
    if os.path.exists(yol):
        print("  var   :", hedef); return
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    print("  indir :", hedef, flush=True)
    istek = urllib.request.Request(url, headers={"User-Agent": "ev-charging-access-tr (research reproduction)"})
    with urllib.request.urlopen(istek, timeout=600) as r, open(yol + ".part", "wb") as f:
        while True:
            parca = r.read(1 << 20)
            if not parca:
                break
            f.write(parca)
    os.replace(yol + ".part", yol)


def main():
    os.makedirs(V, exist_ok=True)
    for url, hedef in DOSYALAR:
        indir(url, hedef)
    # EPDK monthly reports: document ids listed in epdk_arsiv_liste.json (archived from the EPDK statistics page)
    liste = json.load(open(os.path.join(V, "epdk_arsiv_liste.json"), encoding="utf-8"))
    # two September titles are truncated in the EPDK archive ("...-Eyl"); identified by content number
    KESIK = {"34598": "2025-09", "33939": "2024-09"}
    for baslik, no, baglantilar in liste:
        m = re.search(r"-\s*(\w+)\s+(\d{4})", baslik)
        if m and m.group(1) in AY:
            ay = "%s-%02d" % (m.group(2), AY[m.group(1)])
        elif no in KESIK:
            ay = KESIK[no]
        else:
            print("  atlandi (ay okunamadi):", baslik); continue
        hedef = "epdk_aylik/epdk_sarj_%s.pdf" % ay
        indir("https://www.epdk.gov.tr" + baglantilar[0], hedef)


if __name__ == "__main__":
    main()
