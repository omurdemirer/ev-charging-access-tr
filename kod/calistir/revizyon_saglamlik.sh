#!/bin/bash
# Revision robustness analyses (version 1.2.0). Run from the repository root after the main pipeline.
# Network steps are sequential (about 3.5 h with 8 GB RAM); the MILP steps need Gurobi 12.
set -e
cd "$(dirname "$0")/.."
z() { date +%H:%M:%S; }

echo "[$(z)] distance catchment: all load levels, 10-minute indicator, demand allocation variants"
for d in soket ilce_nufus; do for p in 1 5 8; do python faz6_kuyruk_k3.py --pi $p --dagitim $d; done; done
K3_EK=1 K3_R1=1 python faz3_erisim_hesapla.py
K3_EK=1 K3_R1=1 python faz5_enerji_erisim.py --kalibre

echo "[$(z)] alternative catchments with recalibrated energy budgets"
for h in 3mil AC15mil; do
  HAVZA_ALT=$h python faz5_kalibre.py
  HAVZA_ALT=$h python faz5_enerji_erisim.py --kalibre
done

echo "[$(z)] alternative budget definitions (AC,DC kWh)"
for b in "kenarort:1.1133,5.5663" "m10:0.8372,4.7258" "p10:1.0232,5.7760"; do
  BUTCE_ALT=$b python faz5_enerji_erisim.py --kalibre
done

echo "[$(z)] tunnel/bridge correction and vehicle variants"
python faz4t_tunel_kopru.py
for v in "--tunel|_tunel" "--kutle 1196|_m1196" "--kutle 1994|_m1994" "--hizk 0.9|_v0.9" "--hizk 1.1|_v1.1" "--verim 0.782|_e0.782" "--verim 0.874|_e0.874"; do
  f="${v%%|*}"; e="${v##*|}"
  python faz4_enerji_kenar.py $f
  ENERJI_EK="$e" python faz5_kalibre.py
  ENERJI_EK="$e" python faz5_enerji_erisim.py --kalibre
done

echo "[$(z)] winter with 90% and 80% of the summer budget"
for b in "kis90:0.9126447,5.0923588" "kis80:0.8112397,4.5265411"; do
  ENERJI_EK=_paux2.2 BUTCE_ALT=$b python faz5_enerji_erisim.py --kalibre
  python faz7_shapley.py --k3 KIS --enerji "_paux2.2_b${b%%:*}"
done

echo "[$(z)] summaries: tie handling, other inequality measures, energy-distance variants"
python faz5d_esitlik_duyarliligi.py
for k in "" p1.5 p3 p5 p8 p12 w wp1.5 wp3 wp5 wp8 wp12 dsoket p5dsoket p8dsoket dilce_nufus p5dilce_nufus p8dilce_nufus; do
  python faz7b_gostergeler.py --k3 "$k"
done
for h in 3mil AC15mil; do for k in "" p8; do python faz7b_gostergeler.py --havza $h --k3 "$k"; done; done
python faz7d_yuk_ozet.py
python faz7c_varyant_ozet.py

echo "[$(z)] investment model: load assumptions and extended candidate set"
for y in sabit sifir mevcut_oransal; do
  python faz11_optimizasyon.py --butce 100e6 --gap 0.0005 --sure 1800 --yuk $y --noktalar 1,2,12 --ek _yuk$y
  python faz11_optimizasyon.py --butce 100e6 --gap 0.00001 --sure 1800 --yuk $y --hedef9 3.2062473739337582e-06 --ek _yuk$y
done
for e in "" "--aday_ek 20"; do
  python faz11b_asgari_kapsama.py --kapsama_max $e
  for q in 0.8 0.95; do python faz11b_asgari_kapsama.py --kapsama_max --kapsama_hedef $q $e; done
done
echo "[$(z)] done"
