"""EPDK 'Sarj Hizmeti Piyasasi Aylik Istatistikleri' PDF'lerini ayristirir.

Girdi : ../veri/epdk_aylik/epdk_sarj_YYYY-MM.pdf
Cikti : ../veri/epdk_panel_aylik.csv    ulusal aylik panel ('Aylara Gore Sarj Hizmeti Verileri'):
                                        tuketim (MWh), sure (saat), adet, kWh/oturum, saat/oturum
        ../veri/epdk_panel_acdc.csv     raporun ait oldugu ay icin AC/DC kirilimi
        ../veri/epdk_panel_il_top10.csv ay basina ilk 10 il tuketimi ve payi (2025-07'den itibaren)
        ../veri/epdk_panel_revizyon.csv sonraki raporlarda ilk yayindan >%1 sapan degerler
Sayfa numarasina degil etiketlere / ay basliklarina dayanir; bolum yoksa bos birakir.
Her ay icin ILK YAYIMLANDIGI raporun degeri esas alinir (EPDK tablosundaki ezilme hatasi
nedeniyle; asagidaki 'ulusal aylik panel' bolumune bakiniz).
"""
import sys, os, re, glob
import fitz, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
num = lambda s: float(s.replace(".", "").replace(",", ".").replace("%", ""))
NUM = r"-?[\d.]+(?:,\d+)?%?"
AYK = r"(Oca|Şub|Mar|Nis|May|Haz|Tem|Ağu|Eyl|Eki|Kas|Ara)\.(\d{2})"
AYNO = {a: i for i, a in enumerate(["Oca", "Şub", "Mar", "Nis", "May", "Haz",
                                    "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"], 1)}


def lines(page):
    return [x.strip() for x in page.get_text().split("\n") if x.strip()]


def after(L, label, k):
    """label satirindan sonraki ilk k sayisal deger (tek satirda birden cok sayi olabilir)."""
    for i, x in enumerate(L):
        if x.startswith(label):
            vals = []
            for y in L[i + 1:]:
                if not re.fullmatch(r"(?:%s\s*)+" % NUM, y):
                    if vals: break
                    continue
                vals += re.findall(NUM, y)
                if len(vals) >= k: break
            return vals[:k]
    return []


def parse_aylara(L, kaynak):
    """'Aylara Gore' tablosu: ardisik ay basliklari + 5 satir x N ay
    (tuketim, toplam sure, adet, enerji/oturum, sure/oturum). Etiket konumundan bagimsiz."""
    idx = [i for i, x in enumerate(L) if re.fullmatch(AYK, x)]
    if len(idx) < 2:
        return []
    blok = [idx[0]]
    for i in idx[1:]:
        if i != blok[-1] + 1: break
        blok.append(i)
    aylar = []
    for i in blok:
        a, y = re.fullmatch(AYK, L[i]).groups()
        aylar.append("20%s-%02d" % (y, AYNO[a]))
    N, toks = len(aylar), []
    for y in L[blok[-1] + 1:]:
        parts = y.split()
        if parts and all(re.fullmatch(NUM, p) for p in parts):
            toks += parts
        if len(toks) >= 5 * N: break
    if len(toks) < 5 * N:
        return []
    v = [num(t) for t in toks[:5 * N]]
    out = []
    for k, ay in enumerate(aylar):
        mwh = v[k] / 1000 if v[k] > 1e6 else v[k]          # eski raporlar kWh verir
        out.append(dict(ay=ay, kaynak=kaynak, mwh=mwh, saat=v[N + k], adet=v[2 * N + k],
                        kwh_oturum=v[3 * N + k], saat_oturum=v[4 * N + k]))
    return out


aylik, acdc, iller = [], [], []
for f in sorted(glob.glob(os.path.join(D, "epdk_aylik", "epdk_sarj_*.pdf"))):
    ym = re.search(r"\d{4}-\d{2}", f).group(0)
    for p in fitz.open(f):
        L = lines(p)
        txt = " ".join(L)
        if "Aylara Göre" in txt:
            aylik += parse_aylara(L, ym)
        # --- AC/DC kirilimi (raporun ait oldugu ay) ---
        if "Pay Oranı" in txt and "Toplam Şarj Adedi" in txt and "AC" in L:
            row = {"ay": ym}
            for lab, key in [("Toplam Elektrik Tüketimi (kWh)", "kwh"), ("Toplam Şarj Süresi (dakika)", "dk"),
                             ("Toplam Şarj Adedi", "adet")]:
                idx = [i for i, x in enumerate(L) if x.startswith(lab)]
                if idx:
                    v = after(L[idx[-1]:], lab, 4)
                    if len(v) >= 2:
                        row[key + "_ac"], row[key + "_dc"] = num(v[0]), num(v[1])
            if len(row) > 1:
                acdc.append(row)
        # --- ilk 10 il ---
        if "(İlk On İl)" in txt:
            names = [x for x in L if re.fullmatch(r"[A-ZÇĞİÖŞÜ ]{3,}", x) and x not in ("AC", "DC")]
            mwh = [num(x) for x in L if re.fullmatch(r"[\d.]+", x)]
            pay = [num(x) for x in L if x.endswith("%")]
            ax = set(range(0, 100001, 1000))        # eksen etiketlerini ayikla
            mwh = [v for v in mwh if v not in ax or v in mwh[:10]][:10]
            for r, (nm, a, b) in enumerate(zip(names[:10], mwh, pay[:10]), 1):
                iller.append({"ay": ym, "sira": r, "il": nm, "mwh": a, "pay": b})

# --- ulusal aylik panel: ILK YAYIN esas ---
# Gerekce (15.09.2026): EPDK tablosunda sonraki raporlar bazi eski aylarin 'adet' degerini en son
# ayin degeriyle ezmis. Orn. Subat 2026: Sub-Mar-Nis raporlari 2.045.509; Mayis raporu 2.733.420
# (= Mayis degeri); Haz-Tem raporlari 2.713.335 (= Haziran degeri). Oturum basina degerler de ezilmis
# adetten yeniden hesaplandigi icin ic tutarlilik kontrolu bunu yakalamaz. Tuketim/sure revize edilmemis.
A = pd.DataFrame(aylik).sort_values(["ay", "kaynak"])
ilk = A.drop_duplicates("ay", keep="first").set_index("ay")
rev = []
for c in ("mwh", "saat", "adet"):
    d = A.join(ilk[c], on="ay", rsuffix="_ilk")
    d["fark"] = 100 * (d[c] - d[c + "_ilk"]).abs() / d[c + "_ilk"]
    for _, r in d[d.fark > 1].iterrows():
        rev.append(dict(ay=r.ay, alan=c, kaynak=r.kaynak, ilk_yayin=r[c + "_ilk"], deger=r[c], fark_yuzde=round(r.fark, 1)))
rev = pd.DataFrame(rev)
rev.to_csv(os.path.join(D, "epdk_panel_revizyon.csv"), index=False, encoding="utf-8-sig")
P = ilk.reset_index().copy()
# ic tutarlilik: toplamlardan turetilen oturum basina degerler, rapordaki degerlerle uyusmali
P["chk_kwh"] = (1000 * P.mwh / P.adet / P.kwh_oturum - 1).abs().round(3)
P["chk_saat"] = (P.saat / P.adet / P.saat_oturum - 1).abs().round(3)
P["ayni_adet_baska_ay"] = P.adet.duplicated(keep=False)
P.to_csv(os.path.join(D, "epdk_panel_aylik.csv"), index=False, encoding="utf-8-sig")

C = pd.DataFrame(acdc).drop_duplicates("ay").sort_values("ay")
I = pd.DataFrame(iller).drop_duplicates(["ay", "sira"]).sort_values(["ay", "sira"])
C.to_csv(os.path.join(D, "epdk_panel_acdc.csv"), index=False, encoding="utf-8-sig")
I.to_csv(os.path.join(D, "epdk_panel_il_top10.csv"), index=False, encoding="utf-8-sig")

print("ULUSAL AYLIK PANEL: %d ay (%s -> %s)" % (len(P), P.ay.min(), P.ay.max()))
print(P.to_string(index=False))
print("\nTutarlilik ihlali (>%1):", P[(P.chk_kwh > .01) | (P.chk_saat > .01)].ay.tolist())
print("Ayni 'adet' degeri birden cok ayda:", P[P.ayni_adet_baska_ay][["ay", "adet", "kaynak"]].values.tolist())
print("\nRaporlar arasi revizyon (ayni ay, farkli raporlar):")
print(rev.to_string() if len(rev) else "  yok")
print("\nAC/DC kirilimi olan ay:", C.ay.tolist())
print("Ilk-10 il paneli: %d ay (%s -> %s), farkli il: %d" % (I.ay.nunique(), I.ay.min(), I.ay.max(), I.il.nunique()))
