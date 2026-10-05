#!/bin/bash
# revizyon (30.09.2026): enerji v3 + kesin arama siniri + MC tikaniklik
set -e
cd "$(dirname "$0")/.."
zaman() { date +%H:%M:%S; }
echo "[$(zaman)] faz3 (km) K3_EK=1"; K3_EK=1 python faz3_erisim_hesapla.py 2>&1 | tail -3
for v in "|" "--duz|_duz" "--paux 1.2|_paux1.2" "--paux 2.2|_paux2.2" "--regen 0|_regen0" "--regen 0.4|_regen0.4" "--egim 0.15|_egim0.15" "--egim 0.25|_egim0.25"; do
  f="${v%%|*}"; e="${v##*|}"
  echo "[$(zaman)] faz4 $f (ek '$e')"; python faz4_enerji_kenar.py $f 2>&1 | grep -E "Indirgenmis|negatif kenar|Mesafe agirlikli|Cift yonlu"
  echo "[$(zaman)] faz5_kalibre $e"; ENERJI_EK="$e" python faz5_kalibre.py 2>&1 | grep -E "ESLESEN"
  if [ -z "$e" ]; then K=1; else K=0; fi
  echo "[$(zaman)] faz5 $e (K3_EK=$K)"; K3_EK=$K ENERJI_EK="$e" python faz5_enerji_erisim.py --kalibre 2>&1 | grep -E "ENERJIK +(NOM|K1|K1g|K13) " | head -4
done
echo "[$(zaman)] sabit butceli kis"; ENERJI_EK=_paux2.2 ENERJI_BUTCE_EK=_paux1.2 python faz5_enerji_erisim.py --kalibre 2>&1 | grep -E "ENERJIK +(NOM|K1|K13) " | head -3
echo "[$(zaman)] BITTI"
