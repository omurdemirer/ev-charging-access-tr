"""Kultur ve Turizm Bakanligi konaklama verisinden il ve ilce duzeyinde turizm bileseni.

Girdi (../veri/turizm):
  ktb_bakanlik_yillik_2025.xlsx  [Il Ilce]  bakanlik (isletme+basit) belgeli, 2025 yillik, il-ilce
  ktb_bakanlik_aylik_2026-07.xlsx [Il]      bakanlik belgeli, TEMMUZ 2026 (tek ay), il
  ktb_belediye_il_ilce_2022.xlsx  [Il Ilce] belediye (mahalli idare) belgeli, 2022 yillik, il-ilce
Cikti:
  ../veri/turizm/turizm_il.csv, ../veri/turizm/turizm_ilce.csv
Olcu: YERLI geceleme (sarj talebini kendi araciyla gelen yerli turist yaratir; yabanci cogunlukla ucakla gelir).
Varsayim: belediye belgeli 2022 degerleri, bakanlik serisindeki 2022->2025 toplam geceleme buyumesiyle
2025'e tasinir (belediye icin daha yeni il-ilce verisi yayimlanmamis).
Dogrulama (dongusel degil): turizm verisinden yaz katsayisi (il Temmuz payi / yillik pay) ile
EPDK ilk-10 panelinde gozlenen yaz tuketim artisi (Tem-Agu payi / diger aylar) karsilastirilir.
"""
import sys, os
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
T = os.path.join(V, "turizm")
COLS = ["gel_yab", "gel_yerli", "gel_top", "gec_yab", "gec_yerli", "gec_top"]


def tr_upper(s):
    return str(s).strip().replace("i", "İ").replace("ı", "I").upper()


def il_ilce(f):
    df = pd.read_excel(os.path.join(T, f), sheet_name="İl İlçe", header=None, skiprows=3).iloc[:, :8]
    df.columns = ["il", "ilce"] + COLS
    genel = df[df.il.astype(str).str.strip() == "GENEL TOPLAM"]
    df["il"] = df.il.ffill()
    df = df[df.ilce.notna() & (df.il.astype(str).str.strip() != "GENEL TOPLAM")].copy()
    df[COLS] = df[COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["il_ad"] = df.il.map(tr_upper)
    df["ilce"] = df.ilce.astype(str).str.strip()
    ilce = df[df.ilce != "Toplam"][["il_ad", "ilce"] + COLS]
    top = df[df.ilce == "Toplam"].set_index("il_ad")[COLS]
    # Il toplami ILCE satirlarindan hesaplanir: bazi illerde 'Toplam' satiri yok (15.09.2026 bulgusu).
    # 'Toplam' satiri yalnizca kontrol icin kullanilir; ilce satiri olmayan il varsa 'Toplam'dan alinir.
    il = ilce.groupby("il_ad")[COLS].sum().combine_first(top)
    g = float(pd.to_numeric(genel.iloc[0, 6])) if len(genel) else np.nan
    fark_ilce = (il.gec_yerli.reindex(top.index) - top.gec_yerli).abs().max()
    print("%-34s il=%d ilce=%d | yerli geceleme toplami=%s (GENEL=%s) | maks ilce-il farki=%s"
          % (f, len(il), len(ilce), f"{il.gec_yerli.sum():,.0f}", f"{g:,.0f}", f"{fark_ilce:,.0f}"))
    toplamsiz = sorted(set(il.index) - set(top.index))
    if toplamsiz:
        print("   'Toplam' satiri olmayan iller (ilce satirlarindan toplandi): %s" % toplamsiz)
    return il, ilce


bak, bak_ilce = il_ilce("ktb_bakanlik_yillik_2025.xlsx")
bel, bel_ilce = il_ilce("ktb_belediye_il_ilce_2022.xlsx")

# Temmuz 2026 (tek ay) — baslik satiri donemi dogrulamak icin yazdirilir
f_ay = os.path.join(T, "ktb_bakanlik_aylik_2026-07.xlsx")
print("Aylik bulten basligi:", pd.read_excel(f_ay, sheet_name="İl", header=None, nrows=1).iloc[0, 0])
ay = pd.read_excel(f_ay, sheet_name="İl", header=None, skiprows=3).iloc[:, :7]
ay.columns = ["il"] + COLS
ay = ay[ay.il.notna()].copy()
ay[COLS] = ay[COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
ay_top = ay[ay.il.astype(str).str.strip().str.upper() == "TOPLAM"].gec_yerli.sum()
ay = ay[ay.il.astype(str).str.strip().str.upper() != "TOPLAM"]
ay["il_ad"] = ay.il.map(tr_upper)
ay = ay.set_index("il_ad")
print("Temmuz 2026: il=%d | yerli geceleme=%s (TOPLAM satiri=%s)" % (len(ay), f"{ay.gec_yerli.sum():,.0f}", f"{ay_top:,.0f}"))

# belediye 2022 -> 2025 buyume katsayisi (bakanlik toplam geceleme serisi)
y = pd.read_excel(os.path.join(T, "ktb_bakanlik_yillik_2025.xlsx"), sheet_name="Geliş-Geceleme Yıl", header=None)
y = y[pd.to_numeric(y.iloc[:, 0], errors="coerce").notna()].iloc[:, :3]
y.columns = ["yil", "gelis", "geceleme"]
y = y.astype({"yil": int}).set_index("yil")
buyume = float(y.loc[2025, "geceleme"]) / float(y.loc[2022, "geceleme"])
print("Belediye 2022->2025 tasima katsayisi (bakanlik toplam geceleme 2025/2022): %.3f" % buyume)

# il tablosu + TUIK il adlariyla eslesme
pop = pd.read_csv(os.path.join(V, "faz0_ilce.csv")).groupby("il_ad").nufus.sum()
for ad, d in (("bakanlik", bak), ("belediye", bel), ("temmuz", ay)):
    bos = sorted(set(d.index) - set(pop.index))
    if bos: print("  !! %s il adi TUIK ile eslesmeyen: %s" % (ad, bos))
il = pd.DataFrame(index=pop.index)
il["nufus"] = pop
il["gec_yerli_bak_2025"] = bak.gec_yerli.reindex(il.index).fillna(0)
il["gec_yerli_bel_2022"] = bel.gec_yerli.reindex(il.index).fillna(0)
il["gec_yerli_bel_2025tah"] = il.gec_yerli_bel_2022 * buyume
il["gec_yerli_2025tah"] = il.gec_yerli_bak_2025 + il.gec_yerli_bel_2025tah
il["turizm_pay"] = 100 * il.gec_yerli_2025tah / il.gec_yerli_2025tah.sum()
il["turizm_kisi_endeks"] = (il.gec_yerli_2025tah / il.nufus) / (il.gec_yerli_2025tah.sum() / il.nufus.sum())
il["gec_yerli_tem2026"] = ay.gec_yerli.reindex(il.index).fillna(0)
# yaz katsayisi: yalnizca bakanlik belgeli (aylik veri yalnizca onlar icin var)
il["yaz_katsayisi"] = (il.gec_yerli_tem2026 / il.gec_yerli_tem2026.sum()) / (il.gec_yerli_bak_2025 / il.gec_yerli_bak_2025.sum())
il.to_csv(os.path.join(T, "turizm_il.csv"), encoding="utf-8-sig")

ilce = bak_ilce[["il_ad", "ilce", "gec_yerli"]].rename(columns={"gec_yerli": "gec_yerli_bak_2025"}).merge(
    bel_ilce[["il_ad", "ilce", "gec_yerli"]].rename(columns={"gec_yerli": "gec_yerli_bel_2022"}),
    on=["il_ad", "ilce"], how="outer").fillna({"gec_yerli_bak_2025": 0, "gec_yerli_bel_2022": 0})
ilce["gec_yerli_2025tah"] = ilce.gec_yerli_bak_2025 + ilce.gec_yerli_bel_2022 * buyume
ilce.to_csv(os.path.join(T, "turizm_ilce.csv"), index=False, encoding="utf-8-sig")
print("Ilce tablosu: %d satir (TUIK ilce eslesmesi sonraki adimda)" % len(ilce))

print("\nKisi basi yerli turizm endeksi (Turkiye=1), en yuksek 12:")
print(il.sort_values("turizm_kisi_endeks", ascending=False)[["turizm_pay", "turizm_kisi_endeks", "yaz_katsayisi"]].head(12).round(2).to_string())

# --- dogrulama: EPDK'da gozlenen yaz artisi vs turizm yaz katsayisi ---
t = pd.read_csv(os.path.join(V, "epdk_panel_il_top10.csv"))
t["yaz"] = t.ay.str[-2:].isin(["07", "08"])
ye = t.groupby(["il", "yaz"]).pay.mean().unstack().dropna()
ye["epdk_yaz_orani"] = ye[True] / ye[False]
k = ye[["epdk_yaz_orani"]].join(il[["yaz_katsayisi", "turizm_kisi_endeks"]])
print("\nDOGRULAMA — EPDK yaz tuketim artisi vs turizm yaz katsayisi (ilk-10 illeri):")
print(k.sort_values("epdk_yaz_orani", ascending=False).round(2).to_string())
print("Spearman(EPDK yaz orani, turizm yaz katsayisi) = %.2f  (n=%d)"
      % (k.epdk_yaz_orani.corr(k.yaz_katsayisi, method="spearman"), len(k)))
print("Spearman(EPDK yaz orani, kisi basi turizm endeksi) = %.2f"
      % k.epdk_yaz_orani.corr(k.turizm_kisi_endeks, method="spearman"))
