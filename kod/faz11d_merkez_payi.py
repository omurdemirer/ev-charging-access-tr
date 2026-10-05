"""Faz 11d — Pareto noktalarinda merkez butce payinin kayitli kararlardan yeniden hesaplanmasi (revizyon, 05.10.2026).

faz11'in onceki surumu yeni istasyonun merkez payina yalnizca soket maliyetini ekliyor, sabit saha maliyetini
eklemiyordu; payda (toplam harcama) ise sabit maliyeti iceriyordu. Burada optimizasyon yeniden cozulmeden,
faz11_cozum_*.csv karar dosyalarindan pay dogru tanimla yeniden hesaplanir:
    pay = [ sum_merkez FIYAT k (mevcut) + sum_merkez (FIYAT k + SABIT) (yeni) ] / toplam harcama
Merkez: ilce SEGE skorunun ust ucte birlik dilimi (faz11 ile ayni esik).
Once eski tanim (sabit maliyetsiz) ile kayitli degerin birebir uretildigi denetlenir.
Girdi : ../veri/faz11_pareto_<ek>.csv, ../veri/faz11_cozum_<ek>.csv, faz10_indeks.json, faz0_ilce.csv
Cikti : faz11_pareto_<ek>.csv dosyalarinda 'merkez_pay_%' guncellenir, eski deger 'merkez_pay_eski_%' sutununda tutulur.
"""
import json, os, sys
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
IX = json.load(open(os.path.join(V, "faz10_indeks.json"), encoding="utf-8"))
ilce = pd.read_csv(os.path.join(V, "faz0_ilce.csv")).set_index("shapeID")
UST = float(np.nanquantile(ilce.sege_skor.reindex(IX["sid"]).values, 2 / 3))

# (dosya eki, AC fiyat, DC fiyat, sabit maliyet)
SENARYO = [("B100M_siki", 8000, 100000, 50000), ("B25M_siki", 8000, 100000, 50000),
           ("pi5_B100M_siki", 8000, 100000, 50000), ("B100M_K12_K12siki", 8000, 100000, 50000),
           ("B100M_dc60k", 8000, 60000, 50000), ("B100M_ac12k", 12000, 100000, 50000),
           ("B100M_sabit0", 8000, 100000, 0), ("B100M_nexp2400", 8000, 100000, 50000)]

for ek, ac, dc, sabit in SENARYO:
    fiyat = {"AC": ac, "DC": dc}
    P = pd.read_csv(os.path.join(V, "faz11_pareto_%s.csv" % ek))
    K = pd.read_csv(os.path.join(V, "faz11_cozum_%s.csv" % ek))
    eski_kayit = P["merkez_pay_eski_%"] if "merkez_pay_eski_%" in P else P["merkez_pay_%"]
    yeni_pay, eski_pay = [], []
    for i in range(len(P)):
        k = K[K.nokta == i + 1]
        m = k[(k.tur == "mevcut") & (k.sege >= UST)]
        y = k[(k.tur == "yeni") & (k.sege >= UST)]
        soket = (m.k * m.soket.map(fiyat)).sum() + (y.k * y.soket.map(fiyat)).sum()
        saha = sabit * y.kimlik.nunique()
        top = P.butce_mevcut_soket.iloc[i] + P.butce_yeni_istasyon.iloc[i]
        eski_pay.append(100 * soket / max(top, 1))
        yeni_pay.append(100 * (soket + saha) / max(top, 1))
    fark = np.nanmax(np.abs(np.array(eski_pay) - eski_kayit.values))
    print("%-20s eski tanim kayitla uyum: en buyuk fark %.2e | esitlik ucu %.3f -> %.3f"
          % (ek, fark, eski_pay[-1], yeni_pay[-1]))
    assert fark < 1e-6, "eski tanim kayitli degeri uretmiyor: %s" % ek
    P["merkez_pay_eski_%"] = eski_kayit.values
    P["merkez_pay_%"] = yeni_pay
    P.to_csv(os.path.join(V, "faz11_pareto_%s.csv" % ek), index=False, encoding="utf-8-sig")
print("Guncellendi.")
