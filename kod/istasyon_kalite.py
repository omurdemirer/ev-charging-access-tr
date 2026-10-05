"""EPDK istasyon koordinat kalite kontrolu ve duzeltmesi.

Kural
- Koordinatin dustugu il != adres metnindeki il  VE  nokta adres ilinin sinirina > ESIK_KM ise
  koordinat HATALI sayilir; istasyon adres ilcesinin OSM merkezine (ilce_merkez.csv) tasinir
  (ilce eslesmezse o ilin Merkez ilcesine / en kalabalik ilcesine).
- Esik icindeki uyumsuzluklar sinir belirsizligi sayilir; koordinata dokunulmaz.
- Ana kara yol agina kopuk ada ilceleri (Gokceada, Bozcaada, Marmara, Adalar) isaretlenir.
- Il sinirlari geoBoundaries ilce poligonlarinin birlesimidir; mesafeler UTM 36N (EPSG:32636).
Girdi : ../veri/epdk_ulusal_sarj_*.json (en yenisi), ../veri/faz0_ilce.gpkg, ../veri/osm/ilce_merkez.csv
Cikti : ../veri/istasyon_duzeltilmis.csv  (ist_no, hizmet, lat, lon, soket, dc, kw, shapeID, il_ad, ilce_ad,
                                            adres, adres_il, adres_ilce, il_uyumsuz, adres_ile_km,
                                            koordinat_duzeltildi, ada)
"""
import sys, os, re, glob, json, unicodedata
import numpy as np, pandas as pd, geopandas as gpd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
ESIK_KM = 5.0
ADA = {("ÇANAKKALE", "GÖKÇEADA"), ("ÇANAKKALE", "BOZCAADA"), ("BALIKESİR", "MARMARA"), ("İSTANBUL", "ADALAR")}


def n2(s):
    s = str(s).replace("İ", "i").replace("I", "ı").lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).replace("ı", "i")
    return re.sub(r"[^a-z0-9 ]", " ", s)


def sade(s):
    return re.sub(r"\s+", " ", n2(s)).strip()


# --- EPDK ---
f = sorted(glob.glob(os.path.join(V, "epdk_ulusal_sarj_*.json")))[-1]
d = json.load(open(f, encoding="utf-8"))
rows = d.get("data") or d.get("result") or []
if rows and isinstance(rows[0], list):
    rows = [dict(zip(d["columnNames"], r)) for r in rows]
rec = []
for r in rows:
    sk = r.get("soketler") or []
    rec.append(dict(ist_no=r["sarjIstasyonuNo"], hizmet=r["hizmetSekli"], lat=float(r["enlem"]), lon=float(r["boylam"]),
                    soket=len(sk), dc=sum(s["soketTipi"] == "DC" for s in sk),
                    kw=sum(float(s.get("soketGucu") or 0) for s in sk), adres=r.get("adres") or ""))
S = pd.DataFrame(rec)

ilce = gpd.read_file(os.path.join(V, "faz0_ilce.gpkg"))[["shapeID", "il_ad", "ilce_ad", "nufus", "geometry"]].to_crs(32636)
il_poly = ilce.dissolve(by="il_ad")[["geometry"]]
merkez = pd.read_csv(os.path.join(OSM := os.path.join(V, "osm"), "ilce_merkez.csv"))


def ilce_ata(df):
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).to_crs(32636)
    j = gpd.sjoin(g, ilce[["shapeID", "il_ad", "ilce_ad", "geometry"]], how="left", predicate="within").drop(columns="index_right")
    j = j[~j.index.duplicated(keep="first")]
    dis = j.shapeID.isna()
    if dis.any():
        nn = gpd.sjoin_nearest(g[dis], ilce[["shapeID", "il_ad", "ilce_ad", "geometry"]], how="left").drop(columns="index_right")
        nn = nn[~nn.index.duplicated(keep="first")]
        j.loc[dis, ["shapeID", "il_ad", "ilce_ad"]] = nn[["shapeID", "il_ad", "ilce_ad"]]
    return j


# --- adresten il ve ilce ---
iller = sorted(ilce.il_ad.unique(), key=lambda x: -len(x))
il_ilceler = ilce.groupby("il_ad").ilce_ad.apply(lambda s: sorted(set(s), key=lambda x: -len(x))).to_dict()


def adres_coz(a):
    t = sade(a)
    son = [il for il in iller if re.search(r"(^| )" + re.escape(sade(il)) + r"( |$)", t)]
    if not son:
        return None, None
    il = max(son, key=lambda x: t.rfind(sade(x)))
    once = t[: t.rfind(sade(il))].strip()
    ilc = next((c for c in il_ilceler[il] if once.endswith(sade(c))), None)
    return il, ilc


S[["adres_il", "adres_ilce"]] = pd.DataFrame(S.adres.map(adres_coz).tolist(), index=S.index)
J = ilce_ata(S)
J["il_uyumsuz"] = J.adres_il.notna() & (J.adres_il != J.il_ad)
J["adres_ile_km"] = np.nan
u = J.il_uyumsuz
J.loc[u, "adres_ile_km"] = [geom.distance(il_poly.loc[a, "geometry"]) / 1000 for geom, a in zip(J.loc[u, "geometry"], J.loc[u, "adres_il"])]
J["koordinat_duzeltildi"] = u & (J.adres_ile_km > ESIK_KM)

# --- hatali koordinatlari adres ilcesinin OSM merkezine tasi ---
idx = J.index[J.koordinat_duzeltildi]
for i in idx:
    il, ilc = J.at[i, "adres_il"], J.at[i, "adres_ilce"]
    c = merkez[(merkez.il_ad == il) & (merkez.ilce_ad == ilc)] if ilc else merkez.iloc[0:0]
    if c.empty:
        c = merkez[(merkez.il_ad == il) & (merkez.ilce_ad == "MERKEZ")]
    if c.empty:
        c = merkez[merkez.il_ad == il].sort_values("nufus", ascending=False)
    J.at[i, "lon"], J.at[i, "lat"] = float(c.iloc[0].lon), float(c.iloc[0].lat)
if len(idx):
    J2 = ilce_ata(pd.DataFrame(J.loc[idx, ["lon", "lat"]]))
    J.loc[idx, ["shapeID", "il_ad", "ilce_ad"]] = J2[["shapeID", "il_ad", "ilce_ad"]].values
J["ada"] = [(a, b) in ADA for a, b in zip(J.il_ad, J.ilce_ad)]

out = pd.DataFrame(J.drop(columns="geometry"))
out.to_csv(os.path.join(V, "istasyon_duzeltilmis.csv"), index=False, encoding="utf-8-sig")

print("Kaynak:", os.path.basename(f), "| istasyon:", len(out))
print("Adresten il cozulen: %d | ilce cozulen: %d" % (out.adres_il.notna().sum(), out.adres_ilce.notna().sum()))
print("Il uyumsuz: %d | <= %.0f km (sinir belirsizligi, dokunulmadi): %d | > %.0f km (DUZELTILDI): %d (halka acik %d)"
      % (out.il_uyumsuz.sum(), ESIK_KM, (out.il_uyumsuz & ~out.koordinat_duzeltildi).sum(), ESIK_KM,
         out.koordinat_duzeltildi.sum(), (out.koordinat_duzeltildi & (out.hizmet == "HALKA_ACIK")).sum()))
print("Duzeltme sonrasi hala adres iliyle uyumsuz duzeltilmis istasyon:",
      int((out.koordinat_duzeltildi & (out.il_ad != out.adres_il)).sum()))
print("Ada istasyonu: %d (halka acik %d)" % (out.ada.sum(), (out.ada & (out.hizmet == "HALKA_ACIK")).sum()))
print("\nDuzeltilen istasyonlar (adres ili bazinda):")
print(out[out.koordinat_duzeltildi].groupby(["adres_il"]).agg(n=("ist_no", "count"), ort_km=("adres_ile_km", "mean")).round(1).sort_values("n", ascending=False).to_string())
