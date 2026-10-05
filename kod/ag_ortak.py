"""Ortak yol agi fonksiyonlari (faz1c ve sonrasi).

- Ag: faz1_ag_kur.py ciktilarini yukler; en buyuk guclu bagli bilesen uzerinde sure (dk) ve mesafe (km) grafigi.
- dugume_bagla: noktayi en yakin ana bilesen dugumune baglar (baslangic noktalari icin).
- kenara_bagla: noktayi en yakin YOL NOKTASINA ve onun kenarina baglar (istasyonlar icin; otoyol hizmet
  tesisleri gibi dugumlere uzak noktalar dogru baglanir). Kenar yonu (ileri/geri) ve kenar uzerindeki oran saklanir.
- varis: Dijkstra satirlarindan (D: kaynak x dugum) kenar uzerindeki hedefe varis degerini hesaplar.
Yontem ayrintilari icin: faz1_od_matris.py (v2) aciklamasi.
"""
import os
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


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


class Ag:
    def __init__(self, osm_dir, enerji=False):
        A = np.load(os.path.join(osm_dir, "ag.npz"))
        GM = np.load(os.path.join(osm_dir, "ag_geometri.npz"))
        ana = A["dugum_ana"]
        self.yeni = np.full(len(ana), -1, dtype=np.int64); self.yeni[ana] = np.arange(ana.sum())
        self.N = int(ana.sum())
        U0, V0 = A["U"], A["V"]
        m = ana[U0] & ana[V0]
        self.Gt = csr_min(self.yeni[U0[m]], self.yeni[V0[m]], A["dk"][m], self.N)
        self.Gk = csr_min(self.yeni[U0[m]], self.yeni[V0[m]], A["km"][m], self.N)
        self.nlon, self.nlat = A["n_lon"][ana], A["n_lat"][ana]
        if enerji:
            # Enerji grafigi POTANSIYEL DONUSUMLU agirlikla kurulur (w' >= 0, Dijkstra gecerli).
            # Gercek kWh geri kazanimi:  E(o->g) = D'(o->g) - phi[o] + phi[g]
            EN = np.load(os.path.join(osm_dir, "ag_enerji%s.npz" % os.environ.get("ENERJI_EK", "")))
            self.Ge = csr_min(self.yeni[U0[m]], self.yeni[V0[m]], EN["kwh_donusmus"][m], self.N)
            self.phi = EN["phi"][ana]
            self._kwh_yonlu = EN["kwh"]
        self._agac_dugum = cKDTree(duzlem(self.nlon, self.nlat))

        ters = A["ters"].astype(bool)
        F = pd.DataFrame({"pa": A["pos_a"][~ters], "pb": A["pos_b"][~ters], "ga": U0[~ters], "gb": V0[~ters],
                          "dk": A["dk"][~ters], "km": A["km"][~ters], "fwd": True})
        R = pd.DataFrame({"pa": A["pos_a"][ters], "pb": A["pos_b"][ters], "ga": V0[ters], "gb": U0[ters],
                          "dk_r": A["dk"][ters], "km_r": A["km"][ters], "rev": True})
        if enerji:
            # kWh YONE BAGLIDIR: ileri (ga->gb) ve geri (gb->ga) ayri tutulur. dk/km'de bu gerekmez.
            F["kwh"] = self._kwh_yonlu[~ters]
            R["kwh_r"] = self._kwh_yonlu[ters]
        P = F.merge(R, on=["pa", "pb", "ga", "gb"], how="outer")
        P["dk"] = P.dk.fillna(P.dk_r); P["km"] = P.km.fillna(P.km_r)
        P["fwd"] = P.fwd.eq(True); P["rev"] = P.rev.eq(True)
        if enerji:
            P["kwh"] = P.kwh.fillna(0.0); P["kwh_r"] = P.kwh_r.fillna(0.0)
        self._enerji = enerji
        self.P = P[ana[P.ga.values] & ana[P.gb.values]].reset_index(drop=True)

        self.glon, self.glat, wid = GM["lon"], GM["lat"], GM["wid"]
        L = (self.P.pb - self.P.pa + 1).values
        eidx = np.repeat(np.arange(len(self.P)), L)
        pos = np.repeat(self.P.pa.values, L) + (np.arange(L.sum()) - np.repeat(np.cumsum(L) - L, L))
        self._kenar_of = np.full(len(self.glon), -1, dtype=np.int64); self._kenar_of[pos] = eidx
        self._kaps = np.nonzero(self._kenar_of >= 0)[0]
        self._agac_yol = cKDTree(duzlem(self.glon[self._kaps], self.glat[self._kaps]))
        seg = np.where(wid[1:] == wid[:-1], haversine(self.glon[:-1], self.glat[:-1], self.glon[1:], self.glat[1:]), 0.0)
        self._kum = np.r_[0.0, np.cumsum(seg)]

    def dugume_bagla(self, lon, lat):
        _, i = self._agac_dugum.query(duzlem(lon, lat))
        return i, haversine(lon, lat, self.nlon[i], self.nlat[i])

    def kenara_bagla(self, lon, lat):
        _, k = self._agac_yol.query(duzlem(lon, lat))
        p = self._kaps[k]; e = self._kenar_of[p]
        pa, pb = self.P.pa.values[e], self.P.pb.values[e]
        uz = self._kum[pb] - self._kum[pa]
        frac = np.where(uz > 0, (self._kum[p] - self._kum[pa]) / np.where(uz > 0, uz, 1), 0.0).clip(0, 1)
        return {"ga": self.yeni[self.P.ga.values[e]], "gb": self.yeni[self.P.gb.values[e]],
                "fwd": self.P.fwd.values[e], "rev": self.P.rev.values[e], "frac": frac,
                "e_dk": self.P.dk.values[e], "e_km": self.P.km.values[e],
                "e_kwh": self.P.kwh.values[e] if self._enerji else None,
                "e_kwh_r": self.P.kwh_r.values[e] if self._enerji else None,
                "baglanma_km": haversine(lon, lat, self.glon[p], self.glat[p])}

    @staticmethod
    def varis(D, h, w, bag):
        """D: kaynak x dugum Dijkstra; h: kenara_bagla sozlugu (hedefler); w: kenar agirligi (e_dk / e_km); bag: hedef baglanma."""
        ta = np.where(h["fwd"], D[:, h["ga"]] + h["frac"] * w, np.inf)
        tb = np.where(h["rev"], D[:, h["gb"]] + (1 - h["frac"]) * w, np.inf)
        return np.minimum(ta, tb) + bag
