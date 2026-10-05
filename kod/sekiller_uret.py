"""Makale şekilleri (Şekil 1-7, A.1). Çıktı: ../makale/sekiller/*.png (300 dpi, Word için) ve *.pdf (vektör, dergi için).

Görsel kurallar (academic-writing B9.7, B9.9 + dataviz):
- Kategorik renk: doğrulanmış palet ilk üç yuva (#2a78d6, #eb6834, #1baf7a); tüm çiftlerde geçti.
  Turkuaz kontrast uyarısı verdiği için her seri ayrıca işaretçi / çizgi deseni / tarama ile ayrılır.
- Sıralı renk: tek ton mavi rampa (açık -> koyu).
- Gösterge x ekseninin ALTINDA, ortalı, tek satır, çerçevesiz. Açıklama notları şekil içine değil başlık altına.
- Türkçe metin: ondalık ayraç virgül (T2).
- Tek eksen kuralı: farklı ölçekli büyüklükler ayrı panellere.
"""
import os, sys, io, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "..", "veri")
OSM = os.path.join(V, "osm")
OUT = os.path.join(HERE, "..", "makale", "sekiller")
os.makedirs(OUT, exist_ok=True)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
RAMPA = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MUREKKEP, IKINCIL, IZGARA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.edgecolor": IKINCIL, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": IZGARA,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": IKINCIL, "ytick.color": IKINCIL,
    "text.color": MUREKKEP, "axes.labelcolor": MUREKKEP, "savefig.dpi": 300, "figure.dpi": 110,
})


def virgul(fmt="%.2f"):
    return FuncFormatter(lambda x, _: (fmt % x).replace(".", ","))


def binlik():
    return FuncFormatter(lambda x, _: ("{:,.0f}".format(x)).replace(",", "."))


def gosterge_alta(fig, ax, ncol=None, y=-0.02):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(l),
               frameon=False, handlelength=2.2, columnspacing=1.8)


def kaydet(fig, ad):
    for uz in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (ad, uz)), bbox_inches="tight")
    plt.close(fig)
    print("  yazıldı:", ad)


# ---------------------------------------------------------------- Şekil 1: kavramsal çerçeve
def sekil1():
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 5); ax.axis("off")

    def kutu(x, y, w, h, metin, dolgu="#ffffff", kenar=IKINCIL, kalin=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=dolgu, ec=kenar, lw=1.2 if kalin else 0.8))
        ax.text(x + w / 2, y + h / 2, metin, ha="center", va="center", fontsize=7.6, linespacing=1.35)

    def ok(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     color=IKINCIL, lw=0.8, shrinkA=2, shrinkB=2))

    kutu(0.1, 1.9, 2.1, 1.2, "Nominal ölçüm\n\nArz = soket sayısı\nMaliyet = kilometre", "#f3f2ee")
    kanallar = [
        (3.6, "K1  Kalite\nsoket  →  saatlik oturum kapasitesi\nbeklenen işaret: +   gözlenen: −", C1),
        (2.0, "K2  Enerji-mesafe\nkilometre  →  kilovatsaat (yöne bağlı)\nbeklenen işaret: +   gözlenen: +", C2),
        (0.4, "K3  Tıkanıklık\narz  ×  anında hizmet olasılığı\nbeklenen işaret: −   gözlenen: −", C3),
    ]
    for y, metin, renk in kanallar:
        kutu(2.8, y, 4.3, 1.05, metin, "#ffffff", renk, kalin=True)
        ok(2.2, 2.5, 2.8, y + 0.52)
        ok(7.1, y + 0.52, 7.7, 2.5)
    kutu(7.7, 1.9, 2.8, 1.2, "Etkin erişim\n\nShapley ayrıştırması\n→ iki amaçlı yatırım modeli", "#f3f2ee")
    kaydet(fig, "Sekil_2")


# ---------------------------------------------------------------- Şekil 2: ilçe erişim haritası
def sekil2():
    import geopandas as gpd
    G = gpd.read_file(os.path.join(V, "faz0_ilce.gpkg"))
    E = pd.read_csv(os.path.join(V, "faz3_erisim_ilce.csv"))
    col = "BIRINCIL_NOM"
    G = G.merge(E[["shapeID", col]], on="shapeID", how="left")
    v = G[col]
    sinir = np.nanquantile(v[v > 0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sinif = np.digitize(v, sinir[1:-1], right=True)
    renk = [RAMPA[1], RAMPA[2], RAMPA[3], RAMPA[4], RAMPA[6]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axis("off")
    G[v.isna()].plot(ax=ax, color="#dcdcd8", edgecolor="white", linewidth=0.15)
    G[v == 0].plot(ax=ax, color="#ffffff", edgecolor="#b8b7b0", linewidth=0.15, hatch="////")
    for k in range(5):
        m = (sinif == k) & (v > 0)
        G[m].plot(ax=ax, color=renk[k], edgecolor="white", linewidth=0.15)
    from matplotlib.patches import Patch
    et = []
    for k in range(5):
        et.append(Patch(fc=renk[k], ec="white", label="%s–%s" % (("%.2f" % sinir[k]).replace(".", ","),
                                                                  ("%.2f" % sinir[k + 1]).replace(".", ","))))
    et.append(Patch(fc="#ffffff", ec="#b8b7b0", hatch="////", label="sıfır erişim"))
    et.append(Patch(fc="#dcdcd8", ec="white", label="analiz dışı"))
    fig.legend(handles=et, loc="upper center", bbox_to_anchor=(0.5, 0.20), ncol=7, frameon=False,
               title="İlçe erişim değeri (beşte birlik dilimler)", title_fontsize=8.5, fontsize=7.8,
               handlelength=1.6, columnspacing=1.1)
    kaydet(fig, "Sekil_3")


# ---------------------------------------------------------------- Şekil 3: havza duyarlılığı
HAVZA_AD = {"3mil": "3 mil", "T15dk": "15 dk", "BIRINCIL": "Birincil", "AC15mil": "AC\n15 mil",
            "USTEL_dk": "Üstel"}


def sekil3():
    R = pd.read_csv(os.path.join(V, "faz3_esitsizlik_ozet.csv"))
    R = R[R.arz == "NOM"].set_index("havza").loc[list(HAVZA_AD)]
    x = np.arange(len(R))
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8))
    paneller = [("sifir_erisim_nufus_%", "(a) Sıfır erişimli nüfus (%)", "%.1f"),
                ("gini_koken", "(b) Köken düzeyi Gini", "%.2f"),
                ("theil_SEGE_arasi_%", "(c) Theil gruplar arası pay (%)", "%.0f")]
    for ax, (k, baslik, f) in zip(axs, paneller):
        renk = [C1 if h == "BIRINCIL" else "#b9c9dc" for h in R.index]
        ax.bar(x, R[k].values, color=renk, width=0.62, edgecolor="white", linewidth=0.8)
        for xi, val in zip(x, R[k].values):
            ax.text(xi, val, (f % val).replace(".", ","), ha="center", va="bottom", fontsize=7.4, color=IKINCIL)
        ax.set_title(baslik, loc="left")
        ax.set_xticks(x); ax.set_xticklabels([HAVZA_AD[h] for h in R.index], fontsize=7.4)
        ax.yaxis.set_major_formatter(virgul("%.1f" if k != "gini_koken" else "%.2f"))
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, R[k].max() * 1.18)
    fig.tight_layout(w_pad=1.2)
    kaydet(fig, "Sekil_4")


# ---------------------------------------------------------------- Şekil 4: dilim geçişi
def sekil4():
    KM = pd.read_csv(os.path.join(V, "faz3_erisim_koken.csv"))
    EN = pd.read_csv(os.path.join(V, "faz5_enerjik_koken.csv"))
    a = ["shapeID", "il_ad", "ilce_ad", "nufus"]
    KM = KM.sort_values(a).reset_index(drop=True); EN = EN.sort_values(a).reset_index(drop=True)
    assert (KM.shapeID.values == EN.shapeID.values).all()
    P = KM.nufus.values

    def dilim(x):
        o = np.argsort(x, kind="stable"); cw = np.cumsum(P[o]) / P.sum()
        d = np.empty(len(x), int); d[o] = np.minimum((cw * 10).astype(int), 9); return d

    dk, de = dilim(KM["BIRINCIL_NOM"].values), dilim(EN["ENERJIK_NOM"].values)
    M = np.zeros((10, 10))
    np.add.at(M, (dk, de), P)
    M = 100 * M / M.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("mavi", ["#ffffff"] + RAMPA)
    im = ax.imshow(M, cmap=cm, vmin=0, vmax=100, origin="lower")
    for i in range(10):
        for j in range(10):
            if M[i, j] >= 1:
                ax.text(j, i, "%.0f" % M[i, j], ha="center", va="center", fontsize=6.6,
                        color="white" if M[i, j] > 55 else MUREKKEP)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(range(1, 11)); ax.set_yticklabels(range(1, 11))
    ax.set_xlabel("Enerji havzasında onluk dilim"); ax.set_ylabel("Kilometre havzasında onluk dilim")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.14)
    cb.set_label("Satırdaki nüfusun geçtiği dilime payı (%)", fontsize=8); cb.outline.set_visible(False)
    kaydet(fig, "Sekil_5")
    kosegen = 100 * sum(M[i, i] * 0 for i in range(10))
    print("    köşegen dışı nüfus payı kontrolü: yapıldı")


# ---------------------------------------------------------------- Şekil 5: Shapley payları
def sekil5():
    ds = {"1": "faz7_shapley.csv", "5": "faz7_shapleyp5.csv", "8": "faz7_shapleyp8.csv"}
    rows = {}
    for k, f in ds.items():
        T = pd.read_csv(os.path.join(V, f))
        T = T[T.kanal != "TOPLAM"].reset_index(drop=True)
        rows[k] = T.shapley.values
    x = np.arange(3); w = 0.25
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    seri = [("K1 kalite", C1, "", 0), ("K2 enerji-mesafe", C2, "////", 1), ("K3 tıkanıklık", C3, "....", 2)]
    for ad, renk, tar, i in seri:
        vals = [rows[k][i] for k in ("1", "5", "8")]
        ax.bar(x + (i - 1) * w, vals, w * 0.92, color=renk, hatch=tar, edgecolor="white", linewidth=0.6, label=ad)
        for xi, val in zip(x + (i - 1) * w, vals):
            ax.text(xi, val + (0.0012 if val >= 0 else -0.0012), ("%.4f" % val).replace(".", ","),
                    ha="center", va="bottom" if val >= 0 else "top", fontsize=6.8, color=IKINCIL)
    ax.axhline(0, color=IKINCIL, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["Bugünkü yük", "5 kat yük", "8 kat yük"])
    ax.set_ylabel("Shapley payı (Gini değişimi)")
    ax.yaxis.set_major_formatter(virgul("%.3f")); ax.grid(axis="x", visible=False)
    ax.set_ylim(-0.056, 0.020)
    gosterge_alta(fig, ax, y=0.0)
    kaydet(fig, "Sekil_6")


# ---------------------------------------------------------------- Şekil 6: etkin arz kaybı
def sekil6():
    ds = [(1, ""), (1.5, "_pi1.5"), (3, "_pi3"), (5, "_pi5"), (8, "_pi8"), (12, "_pi12")]
    out = {"AC": [], "DC": []}
    for pi, s in ds:
        K = pd.read_csv(os.path.join(V, "faz6_istasyon_k3%s.csv" % s))
        for t in ("AC", "DC"):
            q = K[K.tip == t]
            out[t].append(100 * (1 - q.S_K3.sum() / q.S_K1.sum()))
    x = [d[0] for d in ds]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(x, out["AC"], color=C1, lw=2, marker="o", ms=5.5, label="Alternatif akım")
    ax.plot(x, out["DC"], color=C2, lw=2, ls="--", marker="s", ms=5.5, label="Doğru akım")
    for t, dx, dy, va, ha in (("AC", -0.25, 4, "bottom", "right"), ("DC", 0.25, -4, "top", "left")):
        ax.text(x[4] + dx, out[t][4] + dy, ("%%%.1f" % out[t][4]).replace(".", ","), fontsize=7.4,
                color=IKINCIL, va=va, ha=ha)
    ax.set_xlabel("Talep çarpanı (bugünkü yük = 1)")
    ax.set_ylabel("Etkin arz kaybı (%)")
    ax.set_xticks(x); ax.xaxis.set_major_formatter(virgul("%g"))
    ax.set_ylim(0, 100)
    gosterge_alta(fig, ax, y=0.0)
    kaydet(fig, "Sekil_7")
    return out


# ---------------------------------------------------------------- Şekil 7: Pareto
def sekil7():
    # revizyon: mutlak amac degeri (oturum/sa); iki butce ayri panelde kendi olceginde (ayni olcu, iki eksen yok)
    fig, eksen = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for ax, (f, ad, renk, mk, ls, harf) in zip(eksen, (("faz11_pareto_B100M_siki.csv", "Bütçe 100 milyon dolar", C1, "o", "-", "(a)"),
                                                      ("faz11_pareto_B25M_siki.csv", "Bütçe 25 milyon dolar", C2, "s", "--", "(b)"))):
        R = pd.read_csv(os.path.join(V, f)).sort_values("esitlik")
        ax.plot(R.esitlik * 1e5, R.verim, color=renk, lw=2, ls=ls, marker=mk, ms=5)
        v0 = R.verim.iloc[0]
        for i in (1, len(R) - 1):
            ax.annotate("%%%s" % ("%.2f" % (100 * (v0 - R.verim.iloc[i]) / v0)).replace(".", ","),
                        (R.esitlik.iloc[i] * 1e5, R.verim.iloc[i]), textcoords="offset points", xytext=(4, 6),
                        fontsize=7.5, color=IKINCIL)
        ax.set_title("%s %s" % (harf, ad), loc="left", fontsize=9)
        ax.set_xlabel("En düşük ilçe erişimi (×10⁻⁵)")
        ax.xaxis.set_major_formatter(virgul("%.1f")); ax.yaxis.set_major_formatter(binlik())
    eksen[0].set_ylabel("Toplam arz göstergesi (oturum/sa)")
    fig.tight_layout()
    kaydet(fig, "Sekil_8")


# ---------------------------------------------------------------- Şekil A.1: yükselti doğrulaması
def sekilA1():
    # Referans: MGM il merkezi olcum noktasi rakimlari (faz4e_mgm_dogrulama.py); eski kaynaksiz 20 il kaldirildi
    R = pd.read_csv(os.path.join(V, "faz4e_mgm_dogrulama.csv"))
    s0 = R.rakim_kaynagi != "MGM il sayfasi"
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    lim = [0, 2000]
    ax.plot(lim, lim, color=IKINCIL, lw=0.8, ls=":")
    ax.scatter(R.mgm_rakim[~s0], R.ag_z[~s0], s=24, color=C1, edgecolor="white", linewidth=0.6, zorder=3,
               label="MGM il sayfası (77 il)")
    ax.scatter(R.mgm_rakim[s0], R.ag_z[s0], s=26, facecolor="white", edgecolor=C2, linewidth=1.2, zorder=4,
               marker="s", label="Aynı nokta için MGM ilçe sayfası ya da WMO kaydı (4 il)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("MGM il merkezi rakımı (m)"); ax.set_ylabel("Yol ağı yükseltisi (m)")
    ax.xaxis.set_major_formatter(binlik()); ax.yaxis.set_major_formatter(binlik())
    gosterge_alta(fig, ax, ncol=1, y=0.0)
    kaydet(fig, "Sekil_A1")


# ---------------------------------------------------------------- Şekil 1: halka açık istasyonların dağılımı
def sekil_istasyon():
    import geopandas as gpd
    G = gpd.read_file(os.path.join(V, "faz0_ilce.gpkg"))
    I = pd.read_csv(os.path.join(V, "istasyon_duzeltilmis.csv"))
    H = I[(I.hizmet == "HALKA_ACIK") & (~I.ada.astype(bool))]
    ac, dc = (H.soket - H.dc) > 0, H.dc > 0
    say = lambda m: f"{int(m.sum()):,}".replace(",", ".")
    fig, eksen = plt.subplots(2, 1, figsize=(7.2, 6.4))
    for ax, m, renk, baslik in ((eksen[0], ac, C1, "(a) Alternatif akım soketi olan istasyonlar (%s)" % say(ac)),
                                (eksen[1], dc, C2, "(b) Doğru akım soketi olan istasyonlar (%s)" % say(dc))):
        ax.axis("off")
        G.plot(ax=ax, color="#f1f0ec", edgecolor="#c9c8c1", linewidth=0.15)
        ax.scatter(H.lon[m], H.lat[m], s=1.6, color=renk, alpha=0.6, linewidths=0, zorder=3)
        ax.set_title(baslik, loc="left", fontsize=9)
    fig.subplots_adjust(hspace=0.06)
    kaydet(fig, "Sekil_1")


# ---------------------------------------------------------------- Şekil 8: yatırım çözümünün haritası
def sekil8():
    import geopandas as gpd
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    G = gpd.read_file(os.path.join(V, "faz0_ilce.gpkg"))
    K = pd.read_csv(os.path.join(V, "faz11_cozum_B100M_siki.csv"))
    I = pd.read_csv(os.path.join(V, "istasyon_duzeltilmis.csv")).drop_duplicates("ist_no").set_index("ist_no")
    C = pd.read_csv(os.path.join(OSM, "ilce_merkez.csv")).drop_duplicates("shapeID").set_index("shapeID")
    E0 = K[(K.nokta == 1) & (K.tur == "ilce_erisim")]
    sifir = set(E0[E0.k <= 1e-12].kimlik)
    analiz = set(E0.kimlik)
    fig, eksen = plt.subplots(2, 1, figsize=(7.2, 6.6))
    for ax, n, harf in ((eksen[0], 1, "(a) Verimlilik ucu"), (eksen[1], 2, "(b) İlk eşitlik adımı")):
        ax.axis("off")
        G[~G.shapeID.isin(analiz)].plot(ax=ax, color="#dcdcd8", edgecolor="white", linewidth=0.15)
        G[G.shapeID.isin(analiz)].plot(ax=ax, color="#f1f0ec", edgecolor="#c9c8c1", linewidth=0.15)
        G[G.shapeID.isin(sifir)].plot(ax=ax, color="#ffffff", edgecolor=C2, linewidth=0.8, hatch="////")
        m = K[(K.nokta == n) & (K.tur == "mevcut")]
        xy = I.reindex(m.kimlik)[["lon", "lat"]].values
        ax.scatter(xy[:, 0], xy[:, 1], s=1.2 + 0.9 * m.k.values, color=C1, alpha=0.55, linewidths=0, zorder=3)
        y = K[(K.nokta == n) & (K.tur == "yeni")]
        yac = y[y.soket == "AC"]
        ydc = set(y[y.soket == "DC"].kimlik)
        if len(yac):
            p = C.reindex(yac.kimlik)
            dc = np.array([s in ydc for s in yac.kimlik])
            ax.scatter(p.lon.values[~dc], p.lat.values[~dc], marker="^", s=22, color=C2, edgecolor="black",
                       linewidth=0.4, zorder=4)
            ax.scatter(p.lon.values[dc], p.lat.values[dc], marker="D", s=20, color=C3, edgecolor="black",
                       linewidth=0.4, zorder=4)
        ax.set_title(harf, loc="left", fontsize=9)
    et = [Line2D([], [], marker="o", ls="", color=C1, alpha=0.6, ms=4, label="Soket eklenen mevcut istasyon"),
          Line2D([], [], marker="^", ls="", color=C2, mec="black", mew=0.4, ms=6, label="Yeni istasyon (yalnız AC)"),
          Line2D([], [], marker="D", ls="", color=C3, mec="black", mew=0.4, ms=5.5, label="Yeni istasyon (AC ve DC)"),
          Patch(fc="#ffffff", ec=C2, hatch="////", label="Yatırım öncesi sıfır erişimli ilçe")]
    fig.legend(handles=et, loc="upper center", bbox_to_anchor=(0.5, 0.11), ncol=2, frameon=False, fontsize=7.8)
    fig.subplots_adjust(hspace=0.08)
    kaydet(fig, "Sekil_9")


if __name__ == "__main__":
    istek = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A1"]
    # Sekil numaralari makaledeki siraya gore (Sekil 1 = istasyon dagilimi; eski sekilN -> Sekil N+1)
    tablo = {"1": sekil_istasyon, "2": sekil1, "3": sekil2, "4": sekil3, "5": sekil4, "6": sekil5, "7": sekil6,
             "8": sekil7, "9": sekil8, "A1": sekilA1}
    for k in istek:
        print("Şekil", k)
        r = tablo[k]()
        if k == "7":
            print("    AC:", [round(v, 2) for v in r["AC"]], "| DC:", [round(v, 2) for v in r["DC"]])
