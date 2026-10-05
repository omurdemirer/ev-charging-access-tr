"""Faz 0 — ilce duzeyinde birlesik veri katmani.

Girdiler (../veri):
  epdk_ulusal_sarj_*.json          EPDK ulusal istasyon + soket listesi (en yenisi)
  sege2022_ilce.csv                SEGE-2022 (PDF'ten ayristirilmis; 973 ilce)
  tuik_ilce_nufus.xlsx             TUIK ADNKS 31.12.2021: resmi il/ilce adlari + ilce kodu
  Il ve Ilcelere Gore ... (TR,DF_ADNKS_T22,1.1).csv   TUIK ADNKS 31.12.2025 ilce nufusu (koda gore)
  geoBoundaries-TUR-ADM1/ADM2.geojson   il / ilce sinirlari (CC BY)
Cikti:
  ../veri/faz0_ilce.gpkg ve ../veri/faz0_ilce.csv
"""
import sys, os, re, glob, json, unicodedata
import numpy as np, pandas as pd, geopandas as gpd

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
P = lambda f: os.path.join(V, f)


def n(x):
    """Eslestirme anahtari. PDF 'i' harfini dusurdugu icin i/ı tamamen atilir."""
    x = str(x).replace("İ", "i").replace("I", "ı").lower()
    x = "".join(c for c in unicodedata.normalize("NFKD", x) if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", x.replace("ı", "").replace("i", ""))


def gini(x):
    x = np.sort(np.asarray(x, float))
    k = np.arange(1, len(x) + 1)
    return (2 * (k * x).sum() / (len(x) * x.sum())) - (len(x) + 1) / len(x)


def gini_w(x, w):
    """Nufus agirlikli Gini (Lorenz egrisi alani)."""
    o = np.argsort(x); x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    cw, cxw = np.cumsum(w) / w.sum(), np.cumsum(x * w) / (x * w).sum()
    cw, cxw = np.r_[0, cw], np.r_[0, cxw]
    return 1 - np.sum((cw[1:] - cw[:-1]) * (cxw[1:] + cxw[:-1]))


# --- 1. SEGE: PDF bozulmalarini duzelt -------------------------------------
# PDF fontu bazi harfleri Unicode ozel kullanim alanina (PUA) koyuyor:
# U+E019 = "fl" bitisik harfi; digerleri (orn. U+E01E) i turevleri -> silinir.
def pdf_temizle(x):
    return re.sub("[-]", "", str(x).replace("", "fl"))


s = pd.read_csv(P("sege2022_ilce.csv"))
s["il"], s["ilce"] = s.il.map(pdf_temizle), s.ilce.map(pdf_temizle)
s["ilce"] = s.ilce.replace({"Ondokuzmayıs": "19 Mayıs"})
s["k"] = s.il.map(n) + "|" + s.ilce.map(n)

# --- 2. Nufus (resmi adlar 2021 dosyasindan; nufus 31.12.2025 ADNKS) -------
# 2021 xlsx: resmi il/ilce adlari + 'ILCE KAYIT NO'. 2025 CSV (TUIK veri portali, SDMX):
# IKAMET_YERI = _T (Turkiye) | TRxxx (il, NUTS-3) | 4 haneli ilce kodu (= ILCE KAYIT NO).
# Eslestirme ada gore degil KODA gore yapilir (birebir; 'Merkez' gibi tekrar eden adlardan etkilenmez).
ADNKS_2025 = "İl ve İlçelere Göre İl_İlçe Merkezi, Belde_Köy Nüfusu (TR,DF_ADNKS_T22,1.1).csv"
p = pd.read_excel(P("tuik_ilce_nufus.xlsx"), sheet_name="İLÇE NÜFUSU", header=None, skiprows=8)
p = p[[1, 2, 3, 4]]; p.columns = ["ilce_kod", "il_ad", "ilce_ad", "nufus_2021"]
p = p[pd.to_numeric(p.nufus_2021, errors="coerce").notna()].astype({"ilce_kod": int, "nufus_2021": int})
a = pd.read_csv(P(ADNKS_2025), sep=";", dtype=str, encoding="utf-8-sig")
a = a[(a.YERLESIM_YERI_TUR == "_T") & (a["Zaman (TIME_PERIOD)"] == "2025") & a.IKAMET_YERI.str.fullmatch(r"\d{4}")]
a = a.assign(ilce_kod=a.IKAMET_YERI.astype(int), nufus=a["Gözlem Değeri"].astype(int))[["ilce_kod", "nufus"]]
assert len(a) == 973 and a.nufus.sum() == 86092168, (len(a), a.nufus.sum())
p = p.merge(a, on="ilce_kod", how="left", validate="1:1")
assert p.nufus.notna().all(), p[p.nufus.isna()][["ilce_kod", "il_ad", "ilce_ad"]]
p["k"] = p.il_ad.map(n) + "|" + p.ilce_ad.map(n)
t = s.merge(p, on="k", how="left", validate="1:1")
assert t.nufus.notna().all(), t[t.nufus.isna()][["il", "ilce"]]

# --- 3. Ilce sinirlari -------------------------------------------------------
# geoBoundaries ad hatalari / Ingilizce adlar -> resmi TUIK adi
ALIAS = {"Imbros": "Gökçeada", "Prince Islands": "Adalar", "Karakeçeli": "Karakeçili",
         "Gediz Merkez": "Gediz", "Giresun District": "Merkez"}
a2 = gpd.read_file(P("geoBoundaries-TUR-ADM2.geojson"))[["shapeID", "shapeName", "geometry"]]
a1 = gpd.read_file(P("geoBoundaries-TUR-ADM1.geojson"))[["shapeName", "geometry"]].rename(columns={"shapeName": "il_gb"})
pt = a2.copy(); pt["geometry"] = a2.to_crs(5637).representative_point().to_crs(4326)
pt = gpd.sjoin(pt, a1, how="left", predicate="within")[["shapeID", "shapeName", "il_gb"]]


def pkey(r):
    c = re.sub(r"\(merkez( ilçe)?\)|\bmerkezi?\b", "", ALIAS.get(r.shapeName, r.shapeName), flags=re.I).strip()
    return n(r.il_gb) + "|" + n("Merkez" if (c == "" or n(c) == n(r.il_gb)) else c)


pt["k"] = pt.apply(pkey, axis=1)
# Sinir eslesmesi TUIK'in resmi adlariyla yapilir (SEGE adlari PDF'ten bozuk gelebiliyor)
t["k"] = t.il_ad.map(n) + "|" + t.ilce_ad.map(n)
t = t.merge(pt[["k", "shapeID"]], on="k", how="left", validate="1:1")
assert t.shapeID.notna().all(), t[t.shapeID.isna()][["il_ad", "ilce_ad"]]
g = a2.merge(t[["shapeID", "sira", "il_ad", "ilce_ad", "skor", "nufus"]], on="shapeID", validate="1:1")
g = g.rename(columns={"sira": "sege_sira", "skor": "sege_skor"})

# --- 4. Istasyonlar -> ilce (poligon disindakiler en yakin ilceye) -----------
# 16.09.2026: istasyonlar koordinat kalite kontrolunden gecmis dosyadan okunur (kod/istasyon_kalite.py):
# adres ili ile koordinat ili uyumsuz ve adres iline > 5 km olan 54 istasyon adres ilcesinin merkezine
# tasinmis; ilce atamasi (poligon ici, disindakiler en yakin ilce) o betikte yapilmistir.
f = P("istasyon_duzeltilmis.csv")
j = pd.read_csv(f)
out = j.koordinat_duzeltildi.astype(bool)
j.to_csv(P("faz0_istasyon.csv"), index=False, encoding="utf-8-sig")

pub = j[j.hizmet == "HALKA_ACIK"]
agg = pub.groupby("shapeID").agg(istasyon=("ist_no", "count"), soket=("soket", "sum"), dc=("dc", "sum"), kw=("kw", "sum"))
g = g.merge(agg, on="shapeID", how="left")
g[["istasyon", "soket", "dc", "kw"]] = g[["istasyon", "soket", "dc", "kw"]].fillna(0)
for c in ("istasyon", "soket", "kw"):
    g[c + "_100k"] = g[c] / g.nufus * 1e5
g.to_file(P("faz0_ilce.gpkg"), driver="GPKG")
g.drop(columns="geometry").to_csv(P("faz0_ilce.csv"), index=False, encoding="utf-8-sig")

# --- 5. Rapor + Dingil ve digerleri (2026) ile tutarlilik -------------------
print("Kaynak dosya     :", os.path.basename(f))
print("Ilce             :", len(g), "| nufus toplami:", int(g.nufus.sum()))
print("Istasyon (tumu)  :", len(j), "| koordinati duzeltilen (istasyon_kalite.py):", int(out.sum()))
print("Halka acik       : istasyon=%d soket=%d DC=%d" % (len(pub), pub.soket.sum(), pub.dc.sum()))
print("Istasyonsuz ilce :", int((g.istasyon == 0).sum()), "| bu ilcelerdeki nufus: %.1f%%" % (100 * g.loc[g.istasyon == 0, "nufus"].sum() / g.nufus.sum()))

il = g.groupby("il_ad").agg(nufus=("nufus", "sum"), istasyon=("istasyon", "sum"), soket=("soket", "sum"), kw=("kw", "sum"))
il["ist_100k"] = il.istasyon / il.nufus * 1e5
il["soket_100k"] = il.soket / il.nufus * 1e5
print("\nIL DUZEYI (Dingil+2026 karsilastirmasi: 9,0-155,6 nokta/100k, Gini=0,311)")
print("  istasyon/100k : %.1f - %.1f" % (il.ist_100k.min(), il.ist_100k.max()))
print("  soket/100k    : %.1f - %.1f" % (il.soket_100k.min(), il.soket_100k.max()))
print("  Gini (il, nufus agirlikli)  istasyon=%.3f  soket=%.3f  kW=%.3f" % (
    gini_w(il.istasyon / il.nufus, il.nufus), gini_w(il.soket / il.nufus, il.nufus), gini_w(il.kw / il.nufus, il.nufus)))
print("ILCE DUZEYI")
print("  Gini (ilce, nufus agirlikli) istasyon=%.3f  soket=%.3f  kW=%.3f" % (
    gini_w(g.istasyon / g.nufus, g.nufus), gini_w(g.soket / g.nufus, g.nufus), gini_w(g.kw / g.nufus, g.nufus)))
lo = g.sort_values("kw_100k"); half = lo.nufus.cumsum() <= lo.nufus.sum() / 2
print("  en az hizmet alan nufus yarisinin kW payi: %.1f%%" % (100 * lo.loc[half, "kw"].sum() / g.kw.sum()))
print("  SEGE skoru ile Spearman: istasyon/100k=%.2f  kW/100k=%.2f" % (
    g.sege_skor.corr(g.istasyon_100k, method="spearman"), g.sege_skor.corr(g.kw_100k, method="spearman")))
