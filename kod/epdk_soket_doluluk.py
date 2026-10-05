"""EPDK aylik raporlarindan aylik soket sayilarini (toplam / AC / DC) cikarir ve
ulusal ortalama doluluk (rho) panelini hesaplar.

Rapor s.5'teki 'Toplam / AC / DC Sarj Noktasi Sayisi' grafikleri, rapor ayinda biten
13 aylik seri verir. Grafikte ay etiketi yok -> degerler rapor ayindan geriye dogru atanir.

Girdi : ../veri/epdk_aylik/*.pdf, ../veri/epdk_panel_aylik.csv, ../veri/epdk_panel_acdc.csv
Cikti : ../veri/epdk_panel_soket.csv, ../veri/epdk_panel_doluluk.csv
rho_t = toplam sarj suresi (saat) / (ortalama soket sayisi x aydaki saat)
"""
import sys, os, re, glob, calendar
import fitz, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
num = lambda s: float(s.replace(".", "").replace(",", "."))
LAB = ["Toplam Şarj Noktası (Soket) Sayısı", "AC Şarj Noktası Sayısı", "DC Şarj Noktası Sayısı"]
KEY = ["toplam", "ac", "dc"]


def ay_geri(ym, k):
    y, m = map(int, ym.split("-"))
    t = y * 12 + (m - 1) - k
    return "%d-%02d" % (t // 12, t % 12 + 1)


def saat(ym):
    y, m = map(int, ym.split("-"))
    return calendar.monthrange(y, m)[1] * 24


def seri(L):
    """Uc grafigin 13'er degeri. Her grafigin degerleri, kendi etiketinden once gelir;
    ilk 13 sayi seridir, sonrakiler eksen etiketleridir."""
    pos = [next((i for i, x in enumerate(L) if x.startswith(lab)), None) for lab in LAB]
    if None in pos or pos != sorted(pos):
        return None
    out, bas = {}, 0
    for key, p in zip(KEY, pos):
        vals = [num(t) for x in L[bas:p] for t in x.split() if re.fullmatch(r"[\d.]+", t)]
        if len(vals) < 13:
            return None
        out[key] = vals[:13]
        bas = p + 1
    return out


rows = []
for f in sorted(glob.glob(os.path.join(D, "epdk_aylik", "epdk_sarj_*.pdf"))):
    ym = re.search(r"\d{4}-\d{2}", f).group(0)
    for p in fitz.open(f):
        s = seri([x.strip() for x in p.get_text().split("\n") if x.strip()])
        if s:
            for k in range(13):
                rows.append(dict(ay=ay_geri(ym, 12 - k), kaynak=ym, **{c: s[c][k] for c in KEY}))
            break

S = pd.DataFrame(rows).sort_values(["ay", "kaynak"])
rev = S.groupby("ay").toplam.agg(n_rapor="count", fark=lambda x: x.max() - x.min())
rev = rev[rev.fark > 0]
S = S.drop_duplicates("ay", keep="last").reset_index(drop=True)
S["chk_toplam"] = S.toplam - S.ac - S.dc
ardisik = S.ay.map(lambda a: ay_geri(a, -1)).shift(1).eq(S.ay).tolist()   # onceki ay ile ardisik mi
for c in KEY:
    S[c + "_ort"] = (S[c] + S[c].shift(1)) / 2          # ay basi + ay sonu ortalamasi
S.loc[[not a for a in ardisik], [c + "_ort" for c in KEY]] = float("nan")
S.to_csv(os.path.join(D, "epdk_panel_soket.csv"), index=False, encoding="utf-8-sig")

# --- toplam doluluk (19 aylik ulusal panel) ---
P = pd.read_csv(os.path.join(D, "epdk_panel_aylik.csv"))
M = P.merge(S, on="ay", how="left", suffixes=("", "_sok"))
M["rho"] = M.saat / (M.toplam_ort * M.ay.map(saat))
M.to_csv(os.path.join(D, "epdk_panel_doluluk.csv"), index=False, encoding="utf-8-sig")

# --- AC / DC ayri doluluk ve servis suresi (yalnizca kirilimin yayimlandigi aylar) ---
C = pd.read_csv(os.path.join(D, "epdk_panel_acdc.csv")).merge(S, on="ay", how="left")
C["servis_dk_ac"], C["servis_dk_dc"] = C.dk_ac / C.adet_ac, C.dk_dc / C.adet_dc
C["rho_ac"] = C.dk_ac / 60 / (C.ac_ort * C.ay.map(saat))
C["rho_dc"] = C.dk_dc / 60 / (C.dc_ort * C.ay.map(saat))

print("SOKET PANELI: %d ay (%s -> %s)" % (len(S), S.ay.min(), S.ay.max()))
print("  toplam != AC+DC olan ay:", S[S.chk_toplam != 0][["ay", "chk_toplam"]].values.tolist())
print("  ardisik olmayan ay:", S.ay[[not a for a in ardisik]].tolist()[1:])
print("  raporlar arasi soket revizyonu:", rev.reset_index().values.tolist() if len(rev) else "yok")
print("\nULUSAL DOLULUK (toplam):")
print(M[["ay", "mwh", "saat", "adet", "toplam_ort", "rho"]].round({"toplam_ort": 0, "rho": 4}).to_string(index=False))
print("\nAC / DC (kirilimin yayimlandigi aylar):")
print(C[["ay", "servis_dk_ac", "servis_dk_dc", "rho_ac", "rho_dc"]].round(3).to_string(index=False))
print("  ortalama servis suresi: AC %.1f dk | DC %.1f dk" % (C.servis_dk_ac.mean(), C.servis_dk_dc.mean()))
