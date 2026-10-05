"""Faz 1b — ilce baslangic noktalari + istasyonlarin aga baglanmasi + ilce -> istasyon yol suresi/mesafesi.

Girdi : ../veri/osm/ag.npz, ../veri/osm/ag_geometri.npz, ../veri/osm/yer_dugumleri.csv (faz1_ag_kur.py)
        ../veri/faz0_ilce.gpkg (973 ilce), ../veri/istasyon_duzeltilmis.csv (istasyon_kalite.py)
Cikti : ../veri/osm/ilce_merkez.csv        ilce baslangic noktasi, yontem, baglanma, ada isareti
        ../veri/osm/istasyon_dugum.csv     istasyon -> yol kenari uzerindeki nokta, baglanma mesafesi
        ../veri/osm/ilce_istasyon_od.csv   (ana kara ilcesi, ana kara halka acik istasyonu) -> dk, km

Yontem (v2, 16.09.2026)
- Yalnizca agin en buyuk guclu bagli bileseni.
- Ada ilceleri (Gokceada, Bozcaada, Marmara, Adalar) ana kara yol agina kopuk -> ilce ve istasyonlari
  hesaptan cikarilir (betimsel olarak ayrica raporlanir).
- Ilce baslangici: poligon icindeki OSM yer dugumlerinden adi ilce adiyla (Merkez ilcelerde il adiyla)
  eslesen ilki (city > town > borough > suburb); yoksa poligon temsili noktasi. En yakin ag dugumune baglanir.
- ISTASYONLAR KENAR UZERINE baglanir: sadelestirilmis agda otoyol dugumleri yalniz kavsaklarda oldugundan
  (orn. otoyol hizmet tesisleri en yakin dugume km'lerce uzak) en yakin YOL NOKTASI bulunur; varis
  suresi = min(ileri yon: T(a) + oran*kenar, geri yon: T(b) + (1-oran)*kenar), kenar yonu dikkate alinir.
- Baglanma mesafeleri (ilce -> dugum, istasyon -> yol noktasi) BAG_HIZ km/sa ile sureye, aynen km'ye eklenir.
- Sure: sure agirlikli Dijkstra (limit KESIM_DK); mesafe: km agirlikli Dijkstra (limit KESIM_KM).
  Iki olcu farkli guzergahlardan gelebilir. Serbest akis hizlari; kent ici tikaniklik yok (sinirlilik).
"""
import sys, os, re, time, unicodedata
import numpy as np, pandas as pd, geopandas as gpd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
OSM = os.path.join(V, "osm")
KESIM_DK, KESIM_KM, PARTI, BAG_HIZ = 120.0, 150.0, 8, 30.0
RANK = {"city": 0, "town": 1, "borough": 2, "suburb": 3}
ADA = {("ÇANAKKALE", "GÖKÇEADA"), ("ÇANAKKALE", "BOZCAADA"), ("BALIKESİR", "MARMARA"), ("İSTANBUL", "ADALAR")}


def n2(s):
    s = str(s).replace("İ", "i").replace("I", "ı").lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).replace("ı", "i")
    return re.sub(r"[^a-z]", "", s)


def haversine(lo1, la1, lo2, la2):
    lo1, la1, lo2, la2 = map(np.radians, (np.asarray(lo1, float), np.asarray(la1, float), np.asarray(lo2, float), np.asarray(la2, float)))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


def duzlem(lon, lat):
    return np.c_[np.asarray(lon, float) * 111.320 * np.cos(np.radians(39.0)), np.asarray(lat, float) * 110.574]


def csr_min(U, V, w, n):
    o = np.lexsort((w, V, U)); U, V, w = U[o], V[o], w[o]
    keep = np.r_[True, (U[1:] != U[:-1]) | (V[1:] != V[:-1])]
    return csr_matrix((w[keep], (U[keep], V[keep])), shape=(n, n))


t0 = time.time()
A = np.load(os.path.join(OSM, "ag.npz"))
GM = np.load(os.path.join(OSM, "ag_geometri.npz"))
ana = A["dugum_ana"]
yeni = np.full(len(ana), -1, dtype=np.int64); yeni[ana] = np.arange(ana.sum())
N = int(ana.sum())
U0, V0 = A["U"], A["V"]
m = ana[U0] & ana[V0]
Gt = csr_min(yeni[U0[m]], yeni[V0[m]], A["dk"][m], N)
Gk = csr_min(yeni[U0[m]], yeni[V0[m]], A["km"][m], N)
nlon, nlat = A["n_lon"][ana], A["n_lat"][ana]
agac_dugum = cKDTree(duzlem(nlon, nlat))

# --- fiziksel kenarlar (ileri/geri yon bilgisiyle) ---
ters = A["ters"].astype(bool)
F = pd.DataFrame({"pa": A["pos_a"][~ters], "pb": A["pos_b"][~ters], "ga": U0[~ters], "gb": V0[~ters],
                  "dk": A["dk"][~ters], "km": A["km"][~ters], "fwd": True})
R = pd.DataFrame({"pa": A["pos_a"][ters], "pb": A["pos_b"][ters], "ga": V0[ters], "gb": U0[ters],
                  "dk_r": A["dk"][ters], "km_r": A["km"][ters], "rev": True})
P = F.merge(R, on=["pa", "pb", "ga", "gb"], how="outer")
P["dk"] = P.dk.fillna(P.dk_r); P["km"] = P.km.fillna(P.km_r)
P["fwd"] = P.fwd.eq(True); P["rev"] = P.rev.eq(True)
P = P[ana[P.ga.values] & ana[P.gb.values]].reset_index(drop=True)

glon, glat, wid = GM["lon"], GM["lat"], GM["wid"]
L = (P.pb - P.pa + 1).values
eidx = np.repeat(np.arange(len(P)), L)
pos = np.repeat(P.pa.values, L) + (np.arange(L.sum()) - np.repeat(np.cumsum(L) - L, L))
kenar_of = np.full(len(glon), -1, dtype=np.int64); kenar_of[pos] = eidx
kaps = np.nonzero(kenar_of >= 0)[0]
agac_yol = cKDTree(duzlem(glon[kaps], glat[kaps]))
seg = np.where(wid[1:] == wid[:-1], haversine(glon[:-1], glat[:-1], glon[1:], glat[1:]), 0.0)
kum = np.r_[0.0, np.cumsum(seg)]
print("Ag yuklendi: dugum=%d, fiziksel kenar=%d, yol noktasi=%d (%.0f sn)" % (N, len(P), len(kaps), time.time() - t0), flush=True)

# --- ilce baslangic noktalari ---
ilce = gpd.read_file(os.path.join(V, "faz0_ilce.gpkg"))[["shapeID", "il_ad", "ilce_ad", "nufus", "geometry"]].to_crs(4326)
yer = pd.read_csv(os.path.join(OSM, "yer_dugumleri.csv"))
yer = gpd.GeoDataFrame(yer, geometry=gpd.points_from_xy(yer.lon, yer.lat), crs=4326)
yer = gpd.sjoin(yer, ilce[["shapeID", "geometry"]], predicate="within", how="inner").drop(columns="index_right")
yer["k1"], yer["k2"], yer["rank"] = yer.name.map(n2), yer.name_tr.map(n2), yer.place.map(RANK)
kayit = []
for r in ilce.itertuples():
    hedef = n2(r.il_ad if n2(r.ilce_ad) == "merkez" else r.ilce_ad)
    c = yer[(yer.shapeID == r.shapeID) & ((yer.k1 == hedef) | (yer.k2 == hedef))].sort_values("rank")
    if len(c):
        x = c.iloc[0]; kayit.append((r.shapeID, x.lon, x.lat, "osm_" + x.place, x["name"]))
    else:
        p = r.geometry.representative_point(); kayit.append((r.shapeID, p.x, p.y, "poligon_temsili", None))
M0 = ilce.drop(columns="geometry").merge(pd.DataFrame(kayit, columns=["shapeID", "lon", "lat", "yontem", "osm_ad"]), on="shapeID")
_, M0["dugum"] = agac_dugum.query(duzlem(M0.lon, M0.lat))
M0["baglanma_km"] = haversine(M0.lon, M0.lat, nlon[M0.dugum], nlat[M0.dugum])
M0["ada"] = [(a, b) in ADA for a, b in zip(M0.il_ad, M0.ilce_ad)]
M0.to_csv(os.path.join(OSM, "ilce_merkez.csv"), index=False, encoding="utf-8-sig")
MK = M0[~M0.ada].reset_index(drop=True)
print("Ilce: %d (ada %d cikarildi) | yontem: %s | baglanma medyan %.2f, maks %.2f km"
      % (len(M0), M0.ada.sum(), M0.yontem.value_counts().to_dict(), MK.baglanma_km.median(), MK.baglanma_km.max()), flush=True)

# --- istasyonlar: kenar uzerine baglama ---
S0 = pd.read_csv(os.path.join(V, "istasyon_duzeltilmis.csv"))
_, k = agac_yol.query(duzlem(S0.lon, S0.lat))
p = kaps[k]; e = kenar_of[p]
pa, pb = P.pa.values[e], P.pb.values[e]
uz = kum[pb] - kum[pa]
S0["frac"] = np.where(uz > 0, (kum[p] - kum[pa]) / np.where(uz > 0, uz, 1), 0.0).clip(0, 1)
S0["baglanma_km"] = haversine(S0.lon, S0.lat, glon[p], glat[p])
S0["ga"], S0["gb"] = yeni[P.ga.values[e]], yeni[P.gb.values[e]]
S0["fwd"], S0["rev"] = P.fwd.values[e], P.rev.values[e]
S0["e_dk"], S0["e_km"] = P.dk.values[e], P.km.values[e]
S0.to_csv(os.path.join(OSM, "istasyon_dugum.csv"), index=False, encoding="utf-8-sig")
H = S0[(S0.hizmet == "HALKA_ACIK") & ~S0.ada].reset_index(drop=True)
print("Istasyon: %d | kenar uzerine baglanma medyan %.3f km, p95 %.2f, >2 km: %d | ana kara halka acik: %d"
      % (len(S0), S0.baglanma_km.median(), S0.baglanma_km.quantile(.95), (S0.baglanma_km > 2).sum(), len(H)), flush=True)
g_ = S0[(S0.il_ad == "AKSARAY") & (S0.ilce_ad == "GÜLAĞAÇ")]
if len(g_): print("  kontrol Gulagac (otoyol hizmet tesisi): baglanma maks %.3f km" % g_.baglanma_km.max())

hga, hgb, hfwd, hrev, hfr = H.ga.values, H.gb.values, H.fwd.values, H.rev.values, H.frac.values
bag_dk_h, bag_km_h = H.baglanma_km.values / BAG_HIZ * 60, H.baglanma_km.values


def varis(D, w, bag):
    ta = np.where(hfwd, D[:, hga] + hfr * w, np.inf)
    tb = np.where(hrev, D[:, hgb] + (1 - hfr) * w, np.inf)
    return np.minimum(ta, tb) + bag


# --- sehirler arasi akil sagligi kontrolu (dugumden dugume) ---
def orijin(il, ilce_ad):
    x = MK[(MK.il_ad == il) & (MK.ilce_ad == ilce_ad)]
    return int(x.dugum.iloc[0]) if len(x) else None
CIFT = [(("İSTANBUL", "KADIKÖY"), ("ANKARA", "ÇANKAYA"), "~450 km, ~5 sa"),
        (("ANKARA", "ÇANKAYA"), ("BOLU", "MERKEZ"), "~190 km, ~2 sa"),
        (("İZMİR", "KONAK"), ("ANKARA", "ÇANKAYA"), "~590 km, ~6 sa"),
        (("ANKARA", "ÇANKAYA"), ("ERZURUM", "YAKUTİYE"), "~880 km, ~9 sa")]
print("\nSehirler arasi kontrol (ag | beklenen):")
for a_, b_, bek in CIFT:
    oa, ob = orijin(*a_), orijin(*b_)
    if oa is None or ob is None:
        print("  %s -> %s: ilce yok" % (a_, b_)); continue
    print("  %s/%s -> %s/%s: %.0f km, %.1f sa | %s" % (*a_, *b_, dijkstra(Gk, indices=oa, min_only=True)[ob],
                                                     dijkstra(Gt, indices=oa, min_only=True)[ob] / 60, bek), flush=True)

# --- ilce -> istasyon ---
kayit = {"dk": [], "km": []}
for G, ad, kes, w, bag_h in ((Gt, "dk", KESIM_DK, H.e_dk.values, bag_dk_h), (Gk, "km", KESIM_KM, H.e_km.values, bag_km_h)):
    t1 = time.time()
    bag_o = MK.baglanma_km.values / BAG_HIZ * 60 if ad == "dk" else MK.baglanma_km.values
    for s in range(0, len(MK), PARTI):
        sat = np.arange(s, min(s + PARTI, len(MK)))
        D = dijkstra(G, directed=True, indices=MK.dugum.values[sat], limit=kes)
        T = varis(D, w, bag_h) + bag_o[sat][:, None]
        ii, jj = np.nonzero(T <= kes)
        kayit[ad].append(pd.DataFrame({"o": sat[ii], "h": jj, ad: T[ii, jj]}))
    print("  %s tamam (%.0f sn)" % (ad, time.time() - t1), flush=True)
OD = pd.concat(kayit["dk"]).merge(pd.concat(kayit["km"]), on=["o", "h"], how="outer")
OD["kesim_disi"] = False

bos = np.setdiff1d(np.arange(len(MK)), OD[OD.dk.notna()].o.unique())
for o in bos:
    Tt = varis(dijkstra(Gt, indices=[int(MK.dugum[o])]), H.e_dk.values, bag_dk_h)[0] + MK.baglanma_km[o] / BAG_HIZ * 60
    j = int(np.argmin(Tt))
    Tk = varis(dijkstra(Gk, indices=[int(MK.dugum[o])]), H.e_km.values, bag_km_h)[0] + MK.baglanma_km[o]
    OD = pd.concat([OD, pd.DataFrame([{"o": o, "h": j, "dk": Tt[j], "km": Tk[j], "kesim_disi": True}])], ignore_index=True)

OD = OD.join(MK[["shapeID", "il_ad", "ilce_ad", "nufus"]], on="o")
OD["ist_no"] = H.ist_no.values[OD.h.values]
OD = OD.drop(columns=["o", "h"])[["shapeID", "il_ad", "ilce_ad", "nufus", "ist_no", "dk", "km", "kesim_disi"]]
OD.to_csv(os.path.join(OSM, "ilce_istasyon_od.csv"), index=False, encoding="utf-8-sig")

# --- ozet (ana kara) ---
Z = MK.set_index("shapeID").join(OD.groupby("shapeID").dk.min().rename("en_yakin_dk"))
wn = Z.nufus / Z.nufus.sum()
print("\nOD satiri: %d | kesim disi ilce: %d" % (len(OD), len(bos)))
for esik in (15, 30, 60, 120):
    print("  en yakin halka acik istasyon <= %3d dk: ilce %%%.1f | nufus %%%.1f"
          % (esik, 100 * (Z.en_yakin_dk <= esik).mean(), 100 * wn[Z.en_yakin_dk <= esik].sum()))
print("En uzak 8 ilce:"); print(Z.sort_values("en_yakin_dk", ascending=False)[["il_ad", "ilce_ad", "en_yakin_dk"]].head(8).round(0).to_string())
ada_ist = S0[S0.ada & (S0.hizmet == "HALKA_ACIK")].groupby(["il_ad", "ilce_ad"]).size()
print("\nAda ilceleri (betimsel): nufus %s | halka acik istasyon: %s"
      % (M0[M0.ada].set_index("ilce_ad").nufus.to_dict(), ada_ist.to_dict()))
print("Toplam sure: %.0f sn" % (time.time() - t0))
