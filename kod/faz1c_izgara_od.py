"""Faz 1c — nufus izgarasi (WorldPop 2025, 1 km) -> halka acik istasyon yol suresi/mesafesi (ERISIM tarafi).

Karar (16.09.2026): erisim tarafi baslangiclari ilce merkezi yerine NUFUS IZGARASI; birincil havza EV'ye ozgu
(~5 km ≈ 3 mil), duyarlilik 15 dk ve ustel beta. Kisa havzada ilce merkezi baslangici kirsal ilceleri mekanik olarak
cezalandirirdi.

Girdi : ../veri/worldpop/tur_pop_2025_CN_1km_R2025A_UA_v1.tif (WorldPop R2025A, constrained, 1 km, UN ayarli; CC BY 4.0)
        ../veri/faz0_ilce.gpkg (ADNKS 2025 ilce nufusu), ../veri/istasyon_duzeltilmis.csv, ../veri/osm/ag*.npz
Adimlar
 1) Nufuslu hucre (>0) merkezleri; ilceye ata (poligon ici); ada ilceleri (ana kara agina kopuk) cikarilir.
 2) Hucre nufusu ilce icinde ADNKS 2025 toplamina olceklenir. Hucresi olmayan ilce: ilce poligonu temsili noktasinda
    tek sanal hucre (isaretli).
 3) Hucre -> en yakin ana bilesen dugumu; ayni (dugum, ilce) ciftine dusen hucreler tek KOKEN'de birlesir
    (nufus toplanir, baglanma mesafesi nufus agirlikli). Baglanma BAG_HIZ km/sa ile sureye, aynen km'ye eklenir.
 4) Kokenlerden Dijkstra: sure (limit KESIM_DK) ve km (limit KESIM_KM); istasyonlara kenar uzerinden varis.
 5) Cikti (tam mod): ../veri/osm/izgara_od/parca_XXXX.npz (o int32, h int32, dk float32, km float32; kesim disi = NaN)
    + ../veri/osm/izgara_koken.csv + ../veri/osm/izgara_istasyon.csv
Kullanim: python faz1c_izgara_od.py --deneme 300    (ornek kokenle sure/satir/disk tahmini; dosya yazmaz)
          python faz1c_izgara_od.py                 (tam hesap)
"""
import sys, os, time
sys.modules["boto3"] = None     # makinedeki pyOpenSSL/cryptography uyumsuzlugu: rasterio boto3 olmadan calisir
import numpy as np, pandas as pd, geopandas as gpd
import rasterio
from scipy.sparse.csgraph import dijkstra
from ag_ortak import Ag, haversine

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
OSM = os.path.join(V, "osm")
TIF = os.path.join(V, "worldpop", "tur_pop_2025_CN_1km_R2025A_UA_v1.tif")
# 18.09.2026: birincil kanit (Nicholas, Tal & Turrentine 2020) DC oturumlarinin %65'inin evden 25 mil (40 km)
# icinde oldugunu gosterdi; onceki 3 mil / 16,1 km sinirlari bu kanitla desteklenmiyordu -> kesimler buyutuldu.
KESIM_DK, KESIM_KM, BAG_HIZ, PARTI = 45.0, 40.3, 30.0, 16
ADA = {("ÇANAKKALE", "GÖKÇEADA"), ("ÇANAKKALE", "BOZCAADA"), ("BALIKESİR", "MARMARA"), ("İSTANBUL", "ADALAR")}
deneme = int(sys.argv[sys.argv.index("--deneme") + 1]) if "--deneme" in sys.argv else 0
t0 = time.time()

# --- 1-2) izgara hucreleri, ilce atamasi, ADNKS olcekleme ---
with rasterio.open(TIF) as r:
    a = r.read(1)
    ii, jj = np.nonzero(a > 0)
    lon, lat = rasterio.transform.xy(r.transform, ii, jj, offset="center")
    pop = a[ii, jj].astype(float)
ilce = gpd.read_file(os.path.join(V, "faz0_ilce.gpkg"))[["shapeID", "il_ad", "ilce_ad", "nufus", "geometry"]].to_crs(4326)
ilce = ilce[[(x, y) not in ADA for x, y in zip(ilce.il_ad, ilce.ilce_ad)]].reset_index(drop=True)
H = gpd.GeoDataFrame({"pop": pop}, geometry=gpd.points_from_xy(lon, lat), crs=4326)
H = gpd.sjoin(H, ilce[["shapeID", "geometry"]], predicate="within", how="inner").drop(columns="index_right")
H = H[~H.index.duplicated(keep="first")]
top = H.groupby("shapeID")["pop"].sum()
H["nufus"] = H["pop"] * H.shapeID.map(ilce.set_index("shapeID").nufus / top)
eksik = ilce[~ilce.shapeID.isin(top.index)]
if len(eksik):
    rp = eksik.geometry.representative_point()
    H = pd.concat([H, gpd.GeoDataFrame({"pop": 0.0, "shapeID": eksik.shapeID.values, "nufus": eksik.nufus.values, "sanal": True},
                                       geometry=gpd.points_from_xy(rp.x, rp.y), crs=4326)], ignore_index=True)
H["sanal"] = H.get("sanal", False)
H["sanal"] = H["sanal"].fillna(False)
print("Hucre: %d (ilceye dusen, ana kara) | sanal hucre (hucresiz ilce): %d | nufus %.0f (ADNKS ana kara %.0f) | %.0f sn"
      % (len(H), int(H.sanal.sum()), H.nufus.sum(), ilce.nufus.sum(), time.time() - t0), flush=True)

# --- 3) ag ve kokenler ---
ag = Ag(OSM)
H["dugum"], H["bag_km"] = ag.dugume_bagla(H.geometry.x.values, H.geometry.y.values)
H["w_bag"] = H.nufus * H.bag_km
K = H.groupby(["dugum", "shapeID"], as_index=False).agg(nufus=("nufus", "sum"), w_bag=("w_bag", "sum"), hucre=("pop", "size"))
K["baglanma_km"] = np.where(K.nufus > 0, K.w_bag / K.nufus, 0.0)
K = K.drop(columns="w_bag").merge(ilce[["shapeID", "il_ad", "ilce_ad"]], on="shapeID")
K["lon"], K["lat"] = ag.nlon[K.dugum], ag.nlat[K.dugum]
print("Koken (dugum x ilce): %d | benzersiz dugum: %d | baglanma km (nufus agirlikli) medyan %.2f, p95 %.2f | %.0f sn"
      % (len(K), K.dugum.nunique(), np.median(np.repeat(K.baglanma_km, 1)), K.baglanma_km.quantile(.95), time.time() - t0), flush=True)

# --- istasyonlar ---
S = pd.read_csv(os.path.join(V, "istasyon_duzeltilmis.csv"))
S = S[(S.hizmet == "HALKA_ACIK") & ~S.ada.astype(bool)].reset_index(drop=True)
hs = ag.kenara_bagla(S.lon.values, S.lat.values)
bag_dk_h, bag_km_h = hs["baglanma_km"] / BAG_HIZ * 60, hs["baglanma_km"]

# --- 4) Dijkstra ---
dug = np.unique(K.dugum.values)
rng = np.random.default_rng(20260916)
if deneme:
    dug = rng.choice(dug, size=min(deneme, len(dug)), replace=False)
satir, sure_top = 0, 0.0
if not deneme:
    os.makedirs(os.path.join(OSM, "izgara_od"), exist_ok=True)
    K.to_csv(os.path.join(OSM, "izgara_koken.csv"), index_label="o", encoding="utf-8-sig")
    S[["ist_no", "il_ad", "ilce_ad", "lat", "lon"]].assign(baglanma_km=hs["baglanma_km"]).to_csv(
        os.path.join(OSM, "izgara_istasyon.csv"), index_label="h", encoding="utf-8-sig")
kok_of = K.groupby("dugum").apply(lambda x: x.index.values, include_groups=False)
for pi, s in enumerate(range(0, len(dug), PARTI)):
    grup = dug[s:s + PARTI]
    t1 = time.time()
    Tt = Ag.varis(dijkstra(ag.Gt, indices=grup, limit=KESIM_DK), hs, hs["e_dk"], bag_dk_h)
    Tk = Ag.varis(dijkstra(ag.Gk, indices=grup, limit=KESIM_KM), hs, hs["e_km"], bag_km_h)
    sure_top += time.time() - t1
    oo, hh, dd, kk = [], [], [], []
    for gi, dn in enumerate(grup):
        ok = (Tt[gi] <= KESIM_DK) | (Tk[gi] <= KESIM_KM)
        j = np.nonzero(ok)[0]
        for o in kok_of[dn]:
            b = K.baglanma_km.values[o]
            dkv = Tt[gi, j] + b / BAG_HIZ * 60; kmv = Tk[gi, j] + b
            oo.append(np.full(len(j), o, np.int32)); hh.append(j.astype(np.int32))
            dd.append(np.where(dkv <= KESIM_DK, dkv, np.nan).astype(np.float32))
            kk.append(np.where(kmv <= KESIM_KM, kmv, np.nan).astype(np.float32))
    n = sum(len(x) for x in oo); satir += n
    if not deneme and n:
        np.savez_compressed(os.path.join(OSM, "izgara_od", "parca_%05d.npz" % pi), o=np.concatenate(oo), h=np.concatenate(hh),
                            dk=np.concatenate(dd), km=np.concatenate(kk))
    if pi % 200 == 0:
        print("  parti %d: dugum %d/%d | satir %d | %.0f sn" % (pi, s + len(grup), len(dug), satir, time.time() - t0), flush=True)

if deneme:
    tum = K.dugum.nunique(); oran = tum / len(dug)
    print("\nDENEME (%d rastgele dugum / %d): satir %d -> tum tahmini %.1f milyon satir (~%.2f GB ham) | Dijkstra %.2f sn/dugum -> tum ~%.0f dk"
          % (len(dug), tum, satir, satir * oran / 1e6, satir * oran * 16 / 2 ** 30, sure_top / len(dug), sure_top / len(dug) * tum / 60))
else:
    print("\nTAMAM: satir %d | %.0f sn" % (satir, time.time() - t0))
