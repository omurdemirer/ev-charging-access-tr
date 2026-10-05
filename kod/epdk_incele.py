"""EPDK ulusal sarj verisini inceler: kayit sayisi, kapsam, soket yapisi.
Asil amac: `soketler` alani ANLIK DOLULUK mu veriyor, yoksa statik kapasite mi?

Kullanim:  python epdk_incele.py            (veri/ icindeki en yeni dosyayi alir)
"""
import json, sys, os, glob, collections

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "..", "veri", "epdk_ulusal_sarj_*.json")))
if not files:
    sys.exit("veri/ icinde epdk_ulusal_sarj_*.json yok")
path = files[-1]
d = json.load(open(path, encoding="utf-8"))
rows = d.get("data") or d.get("result") or []
cols = d.get("columnNames") or []
if rows and isinstance(rows[0], list):          # satirlar sutun listesi bicimindeyse
    rows = [dict(zip(cols, r)) for r in rows]
print("Dosya     :", os.path.basename(path))
print("numRows   :", d.get("numRows"), "| result uzunlugu:", len(rows))

# --- Kapsam: ulusal mi? ---
ll = [(r.get("enlem"), r.get("boylam")) for r in rows]
ll = [(float(a), float(b)) for a, b in ll if a not in (None, "") and b not in (None, "")]
if ll:
    la, lo = zip(*ll)
    print("Koordinat : enlem %.2f-%.2f  boylam %.2f-%.2f  (Turkiye ~ 36-42 / 26-45)"
          % (min(la), max(la), min(lo), max(lo)))
print("Koordinatsiz kayit:", len(rows) - len(ll))
print("Hizmet sekli:", dict(collections.Counter(r.get("hizmetSekli") for r in rows)))
print("Isletmeci sayisi:", len({r.get("sarjAgiIsletmecisiLisansNo") for r in rows}))

# --- Soket yapisi: KRITIK SORU ---
sk_all = [s for r in rows for s in (r.get("soketler") or [])]
print("\nToplam soket:", len(sk_all),
      "| soketi bos istasyon:", sum(1 for r in rows if not r.get("soketler")))
if sk_all:
    print("\n>>> SOKET ALANLARI (ilk soket):")
    for k, v in sk_all[0].items():
        print("   %-32s = %r" % (k, v))
    print("\n>>> Her alanin farkli deger sayisi (dusuk sayi = kategorik; durum alani burada gorunur):")
    keys = sorted({k for s in sk_all for k in s})
    for k in keys:
        vals = collections.Counter(str(s.get(k)) for s in sk_all)
        ornek = ", ".join("%s(%d)" % (v[:20], c) for v, c in vals.most_common(5))
        print("   %-32s farkli=%-6d  %s" % (k, len(vals), ornek))
