"""Faz 1a — Turkiye karayolu agi (OSM): yonlu, sadelestirilmis ag.

Girdi : ../veri/osm/turkey-latest.osm.pbf  (Geofabrik; (c) OpenStreetMap katkicilari, ODbL)
Cikti : ../veri/osm/ag.npz             dugum koordinatlari + yonlu kenarlar (km, dk, sinif, geometri indeksi)
        ../veri/osm/ag_geometri.npz    tum yol noktalarinin koordinatlari (K2 topografya icin)
        ../veri/osm/yer_dugumleri.csv  place=city/town/borough/suburb dugumleri (ilce merkezi eslemesi icin)

Notlar
- Windows'ta pyosmium (C++) Turkce karakterli yollari ACAMIYOR (16.09.2026 dogrulandi) ->
  PBF, ASCII yoldaki calisma klasorune (%TEMP%\\ev_osm) kopyalanip oradan okunur.
- Yol siniflari: motorway, trunk, primary, secondary, tertiary (+ _link). Konut ici yollar dahil degil.
- Hiz (km/sa): maxspeed sayisal ise min(maxspeed, sinif ust siniri), degilse sinif varsayilani.
- Tek yon: oneway=yes/true/1 ileri, -1 geri, no cift; motorway, motorway_link ve
  junction=roundabout/circular etiketsizse tek yon (OSM varsayimi).
- Sadelestirme: yalnizca kavsak (>=2 yol ortak dugum) ve yol uc dugumleri ag dugumu olur;
  aradaki zincir tek kenar. Kenar uzunlugu, zincirdeki tum noktalar uzerinden haversine toplamidir.
- Kullanim: python faz1_ag_kur.py            (dugum indeksi bellekte, ~1,5 GB)
            python faz1_ag_kur.py --disk     (bellek yetmezse dugum indeksi diskte, daha yavas)
"""
import sys, os, re, time, shutil
from array import array
import numpy as np
import pandas as pd
import osmium
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OSM = os.path.join(HERE, "..", "veri", "osm")
KAYNAK = os.path.join(OSM, "turkey-latest.osm.pbf")

SINIF = {  # ad: (kod, varsayilan hiz, ust sinir)  km/sa
    "motorway": (1, 110, 120), "motorway_link": (2, 60, 80),
    "trunk": (3, 90, 110), "trunk_link": (4, 50, 70),
    "primary": (5, 75, 90), "primary_link": (6, 45, 60),
    "secondary": (7, 60, 90), "secondary_link": (8, 40, 60),
    "tertiary": (9, 45, 90), "tertiary_link": (10, 30, 50),
}
KOD_AD = {v[0]: k for k, v in SINIF.items()}
TEK_YON_VARSAYILAN = {"motorway", "motorway_link"}
YER = {"city", "town", "borough", "suburb"}


def ascii_calisma_kopyasi():
    tmp = os.environ.get("TEMP") or os.environ.get("TMP") or ""
    d = os.path.join(tmp, "ev_osm")
    if not d.isascii():
        sys.exit("ASCII calisma klasoru yok (TEMP Turkce karakter iceriyor): %s" % d)
    os.makedirs(d, exist_ok=True)
    hedef = os.path.join(d, "turkey.osm.pbf")
    if not (os.path.exists(hedef) and os.path.getsize(hedef) == os.path.getsize(KAYNAK)):
        print("PBF ASCII yola kopyalaniyor:", hedef, flush=True)
        shutil.copyfile(KAYNAK, hedef)
    return d, hedef


def hiz(tag, sinif):
    _, vars_, ust = SINIF[sinif]
    m = re.match(r"^\s*(\d+)", tag or "")
    if m and int(m.group(1)) > 0:
        return float(min(int(m.group(1)), ust))
    return float(vars_)


def haversine(lo1, la1, lo2, la2):
    lo1, la1, lo2, la2 = map(np.radians, (lo1, la1, lo2, la2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


def csr_min(U, V, w, n):
    """Ayni (u,v) cifti icin en kucuk agirlikli kenari tutar (csr toplama yapmasin)."""
    o = np.lexsort((w, V, U))
    U, V, w = U[o], V[o], w[o]
    keep = np.r_[True, (U[1:] != U[:-1]) | (V[1:] != V[:-1])]
    return csr_matrix((w[keep], (U[keep], V[keep])), shape=(n, n))


def main():
    t0 = time.time()
    d, pbf = ascii_calisma_kopyasi()
    indeks = "sparse_file_array," + os.path.join(d, "_dugum_indeksi.bin") if "--disk" in sys.argv else "sparse_mem_array"
    print("Dugum indeksi:", indeks, flush=True)

    ids, lon, lat = array("q"), array("d"), array("d")
    w_bas, w_sinif, w_yon, w_hiz, w_osm = array("q"), array("B"), array("b"), array("f"), array("q")
    yer, atlanan = [], 0
    fp = (osmium.FileProcessor(pbf).with_locations(indeks)
          .with_filter(osmium.filter.KeyFilter("highway", "place")))
    for o in fp:
        if o.is_way():
            hw = o.tags.get("highway")
            if hw not in SINIF or len(o.nodes) < 2:
                continue
            nd = [(n.ref, n.location) for n in o.nodes]
            if not all(loc.valid() for _, loc in nd):
                atlanan += 1
                continue
            w_bas.append(len(ids)); w_sinif.append(SINIF[hw][0]); w_osm.append(o.id)
            ow, jn = o.tags.get("oneway"), o.tags.get("junction")
            if ow in ("yes", "true", "1"): y = 1
            elif ow == "-1": y = -1
            elif ow == "no": y = 0
            elif hw in TEK_YON_VARSAYILAN or jn in ("roundabout", "circular"): y = 1
            else: y = 0
            w_yon.append(y); w_hiz.append(hiz(o.tags.get("maxspeed"), hw))
            for ref, loc in nd:
                ids.append(ref); lon.append(loc.lon); lat.append(loc.lat)
        elif o.is_node():
            p = o.tags.get("place")
            if p in YER and o.location.valid():
                yer.append((o.id, p, o.tags.get("name"), o.tags.get("name:tr"), o.location.lon, o.location.lat))
    print("PBF okundu: %.0f sn | hedef yol=%d | yol noktasi=%d | gecersiz konumlu atlanan yol=%d | yer dugumu=%d"
          % (time.time() - t0, len(w_bas), len(ids), atlanan, len(yer)), flush=True)

    ids = np.frombuffer(ids, dtype=np.int64); lon = np.frombuffer(lon, dtype=np.float64); lat = np.frombuffer(lat, dtype=np.float64)
    w_bas = np.frombuffer(w_bas, dtype=np.int64); w_sinif = np.frombuffer(w_sinif, dtype=np.uint8)
    w_yon = np.frombuffer(w_yon, dtype=np.int8); w_hiz = np.frombuffer(w_hiz, dtype=np.float32)
    w_osm = np.frombuffer(w_osm, dtype=np.int64)
    W, NR = len(w_bas), len(ids)
    w_son = np.r_[w_bas[1:], NR]
    wid = np.repeat(np.arange(W), w_son - w_bas)

    # --- ag dugumleri: kavsaklar + yol uclari ---
    u, ilk, inv, cnt = np.unique(ids, return_index=True, return_inverse=True, return_counts=True)
    bayrak = cnt >= 2
    bayrak[inv[w_bas]] = True
    bayrak[inv[w_son - 1]] = True
    g_u = np.cumsum(bayrak) - 1
    n_lon, n_lat, n_osm = lon[ilk][bayrak], lat[ilk][bayrak], u[bayrak]
    N = int(bayrak.sum())

    # --- kenarlar: ayni yol icinde ardisik ag dugumleri arasi zincirler ---
    ayni = wid[1:] == wid[:-1]
    seg = np.where(ayni, haversine(lon[:-1], lat[:-1], lon[1:], lat[1:]), 0.0)
    kum = np.r_[0.0, np.cumsum(seg)]
    gpos = np.nonzero(bayrak[inv])[0]
    a, b = gpos[:-1], gpos[1:]
    m = wid[a] == wid[b]
    a, b = a[m], b[m]
    km = kum[b] - kum[a]
    ga, gb = g_u[inv[a]], g_u[inv[b]]
    m = (ga != gb) & (km > 0)
    a, b, km, ga, gb = a[m], b[m], km[m], ga[m], gb[m]
    ew = wid[a]
    yon, spd, sn = w_yon[ew], w_hiz[ew].astype(np.float64), w_sinif[ew]
    dk = km / spd * 60.0
    print("Fiziksel kenar (yonsuz)=%d | toplam %.0f km" % (len(km), km.sum()), flush=True)

    ileri, geri = yon >= 0, yon <= 0
    U = np.r_[ga[ileri], gb[geri]]; V = np.r_[gb[ileri], ga[geri]]
    E_km = np.r_[km[ileri], km[geri]]; E_dk = np.r_[dk[ileri], dk[geri]]
    E_sn = np.r_[sn[ileri], sn[geri]]; E_way = np.r_[ew[ileri], ew[geri]]
    E_pa = np.r_[a[ileri], a[geri]]; E_pb = np.r_[b[ileri], b[geri]]
    E_ters = np.r_[np.zeros(ileri.sum(), np.int8), np.ones(geri.sum(), np.int8)]

    # --- en buyuk guclu bagli bilesen ---
    G = csr_min(U, V, E_dk, N)
    ncomp, lab = connected_components(G, directed=True, connection="strong")
    buyuk = np.bincount(lab).argmax()
    dugum_ana = lab == buyuk
    kenar_ana = dugum_ana[U] & dugum_ana[V]
    print("Guclu bagli bilesen: %d | en buyugu dugumlerin %%%.1f'i, yonlu kenar km'sinin %%%.1f'i"
          % (ncomp, 100 * dugum_ana.mean(), 100 * E_km[kenar_ana].sum() / E_km.sum()), flush=True)

    np.savez_compressed(os.path.join(OSM, "ag.npz"),
                        n_lon=n_lon, n_lat=n_lat, n_osm=n_osm, dugum_ana=dugum_ana,
                        U=U, V=V, km=E_km, dk=E_dk, sinif=E_sn, way=E_way, pos_a=E_pa, pos_b=E_pb, ters=E_ters,
                        w_osm=w_osm, w_sinif=w_sinif, w_yon=w_yon, w_hiz=w_hiz)
    np.savez_compressed(os.path.join(OSM, "ag_geometri.npz"), lon=lon, lat=lat, wid=wid)
    pd.DataFrame(yer, columns=["osm_id", "place", "name", "name_tr", "lon", "lat"]).to_csv(
        os.path.join(OSM, "yer_dugumleri.csv"), index=False, encoding="utf-8-sig")

    ozet = pd.DataFrame({"sinif": [KOD_AD[k] for k in sn], "km": km}).groupby("sinif").km.sum().sort_values(ascending=False)
    print("\nSinifa gore fiziksel yol uzunlugu (km, yonsuz, cift yonlu ayrik yollar iki kez sayilabilir):")
    print(ozet.round(0).to_string())
    print("\nAg: dugum=%d | yonlu kenar=%d | ana bilesen dugum=%d | toplam sure %.0f sn"
          % (N, len(U), int(dugum_ana.sum()), time.time() - t0))
    ix = os.path.join(d, "_dugum_indeksi.bin")
    if os.path.exists(ix):
        os.remove(ix)


if __name__ == "__main__":
    main()
