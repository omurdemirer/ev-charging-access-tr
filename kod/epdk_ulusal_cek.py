"""EPDK ulusal sarj istasyonu listesini ceker.

Kaynak : https://apigateway.epdk.gov.tr/sarjIstasyonlari   (GET, govde: {})
Kimlik dogrulama gerekmez ("security": [] -- swagger'dan dogrulandi, 11.09.2026).
KOTA (resmi kilavuz): parametresiz sorgu SAATTE 1, parametreli sorgu DAKIKADA 1.
Kota IP duzeyindedir. Kilavuz: ../veri/EPDK_web_servis_kilavuzu.docx

Kullanim:  python epdk_ulusal_cek.py
Cikti   :  ../veri/epdk_ulusal_sarj_<TARIH>.json
"""
import json, time, sys, os, datetime, urllib.request

URL = "https://apigateway.epdk.gov.tr/sarjIstasyonlari"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "veri")
MAX_TRIES = 16      # ~80 dk
WAIT = 300          # 5 dk


def attempt():
    req = urllib.request.Request(
        URL, data=b"{}", method="GET",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out = os.path.join(OUTDIR, "epdk_ulusal_sarj_%s.json" % stamp)

    for i in range(1, MAX_TRIES + 1):
        try:
            d = attempt()
            n = d.get("numRows")
            if d.get("statusCode") == 200 and n:
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False)
                print("BASARILI (deneme %d): %s kayit" % (i, n))
                print("Dosya: %s" % os.path.normpath(out))
                return 0
            print("deneme %d: statusCode=%s numRows=%s" % (i, d.get("statusCode"), n), flush=True)
        except Exception as e:
            print("deneme %d: %s" % (i, str(e)[:120]), flush=True)
        if i < MAX_TRIES:
            print("   kota bekleniyor, %d dk sonra tekrar..." % (WAIT // 60), flush=True)
            time.sleep(WAIT)

    print("BASARISIZ: %d denemede alinamadi. Daha sonra tekrar calistirin." % MAX_TRIES)
    return 1


if __name__ == "__main__":
    sys.exit(main())
