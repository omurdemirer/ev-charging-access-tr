"""Faz 7b — Gini disindaki esitsizlik gostergeleriyle Shapley ayristirmasi (revizyon).

Her gosterge icin sekiz yapilandirma (faz7 ile ayni sut() tanimi) hesaplanir ve kanal paylari cikarilir.
Gostergeler (nufus agirlikli, koken duzeyi):
  gini      : Gini katsayisi (temel)
  theil     : Theil T (sifirlar icin x ln x -> 0)
  atkinson  : Atkinson, esitsizlik kacinma parametresi 0,5 (sifir degerlerle tanimli)
  cv        : degisim katsayisi
  palma     : en yuksek %10 nufusun erisim payi / en dusuk %40 nufusun erisim payi
  alt20     : en dusuk %20 nufusun erisim payi (%; yukselmesi esitlik lehine)
Girdi : ../veri/faz3_erisim_koken.csv, ../veri/faz5_enerjik_koken.csv
Kullanim: python faz7b_gostergeler.py [--k3 <etiket>] [--havza 3mil|AC15mil] [--enerji_ek <ek>]
Cikti : ../veri/faz7b_gostergeler<etiket>.csv
"""
import os, sys, itertools
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")
_a = lambda k, v: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else v
SON = _a("--k3", "")
# revizyon: --havza 3mil|AC15mil -> km tarafi o havzanin oneki, enerji tarafi faz5_enerjik_h<havza>_koken.csv
#             --enerji_ek <ek>     -> km tarafi BIRINCIL, enerji tarafi faz5_enerjik<ek>_koken.csv (butce/arac varyanti)
HAVZA = _a("--havza", ""); EEK = _a("--enerji_ek", "")
KM_ON = (HAVZA + "_") if HAVZA else "BIRINCIL_"
EN_DOSYA = "faz5_enerjik%s_koken.csv" % (("_h" + HAVZA) if HAVZA else EEK)
CIKTI_EK = SON + (("_h" + HAVZA) if HAVZA else "") + EEK
anah = ["shapeID", "il_ad", "ilce_ad", "nufus"]


def sut(k1, k2, k3):
    return ("ENERJIK_" if k2 else KM_ON) + ("K1" if k1 else "NOM") + (("3" + SON) if k3 else "")


kol = sorted({sut(*c) for c in itertools.product([0, 1], repeat=3)})
KM = pd.read_csv(os.path.join(V, "faz3_erisim_koken.csv"), usecols=anah + [k for k in kol if k.startswith(KM_ON)])
EN = pd.read_csv(os.path.join(V, EN_DOSYA), usecols=anah + [k for k in kol if k.startswith("ENE")])
KM = KM.sort_values(anah).reset_index(drop=True); EN = EN.sort_values(anah).reset_index(drop=True)
assert (KM.shapeID.values == EN.shapeID.values).all()
P = KM.nufus.values.astype(float); w = P / P.sum()


def paylar(x):
    o = np.argsort(x, kind="stable"); cw = np.cumsum(w[o]); cx = np.cumsum((w * x)[o]) / (w * x).sum()
    return cw, cx


def gosterge(x):
    mu = (w * x).sum(); r = x / mu
    cw, cx = paylar(x)
    gini = 1 - np.sum(np.diff(np.r_[0, cw]) * (cx + np.r_[0, cx[:-1]]))
    theil = np.sum(w * np.where(r > 0, r * np.log(np.where(r > 0, r, 1)), 0.0))
    atk = 1 - (np.sum(w * np.sqrt(x))) ** 2 / mu
    cv = np.sqrt(np.sum(w * (x - mu) ** 2)) / mu
    pay = lambda q: np.interp(q, np.r_[0, cw], np.r_[0, cx])
    palma = (1 - pay(0.9)) / pay(0.4)
    alt20 = 100 * pay(0.2)
    return {"gini": gini, "theil": theil, "atkinson": atk, "cv": cv, "palma": palma, "alt20": alt20}


v = {}
for c in itertools.product([0, 1], repeat=3):
    v[c] = gosterge((EN if c[1] else KM)[sut(*c)].values)
fakt = [1, 1, 2, 6]; sat = []
for g in v[(0, 0, 0)]:
    phi = []
    for i in range(3):
        t = 0.0
        for S in itertools.product([0, 1], repeat=3):
            if S[i]:
                continue
            Si = list(S); Si[i] = 1; k = sum(S)
            t += fakt[k] * fakt[2 - k] / 6 * (v[tuple(Si)][g] - v[S][g])
        phi.append(t)
    top = v[(1, 1, 1)][g] - v[(0, 0, 0)][g]
    assert abs(sum(phi) - top) < 1e-9
    mut = sum(abs(p) for p in phi)
    sat.append({"gosterge": g, "nominal": v[(0, 0, 0)][g], "uc_kanal": v[(1, 1, 1)][g],
                "K1": phi[0], "K2": phi[1], "K3": phi[2], "toplam": top,
                "K1_mut_%": 100 * abs(phi[0]) / mut, "K2_mut_%": 100 * abs(phi[1]) / mut, "K3_mut_%": 100 * abs(phi[2]) / mut})
R = pd.DataFrame(sat)
print("K3 etiketi: '%s' | km: %s | enerji: %s" % (SON, KM_ON, EN_DOSYA)); print(R.round(4).to_string(index=False))
R.to_csv(os.path.join(V, "faz7b_gostergeler%s.csv" % CIKTI_EK), index=False, encoding="utf-8-sig")
