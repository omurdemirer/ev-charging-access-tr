#!/bin/bash
# 30.09.2026 durdurulan boru hattinin kalan adimlari (regen0.4, egim 0.15/0.25, sabit butceli kis)
set -e
cd "$(dirname "$0")/.."
zaman() { date +%H:%M:%S; }
for v in "--regen 0.4|_regen0.4" "--egim 0.15|_egim0.15" "--egim 0.25|_egim0.25"; do
  f="${v%%|*}"; e="${v##*|}"
  echo "[$(zaman)] faz4 $f (ek '$e')"; python faz4_enerji_kenar.py $f 2>&1 | grep -E "Indirgenmis|negatif kenar|Mesafe agirlikli|Cift yonlu"
  echo "[$(zaman)] faz5_kalibre $e"; ENERJI_EK="$e" python faz5_kalibre.py 2>&1 | grep -E "ESLESEN"
  echo "[$(zaman)] faz5 $e"; K3_EK=0 ENERJI_EK="$e" python faz5_enerji_erisim.py --kalibre 2>&1 | grep -E "ENERJIK +(NOM|K1|K1g|K13) " | head -4
done
echo "[$(zaman)] faz4 ana (faz4_ebar.json ve ana kenar dosyasini geri yukle)"; python faz4_enerji_kenar.py 2>&1 | grep -E "Mesafe agirlikli"
echo "[$(zaman)] sabit butceli kis"; ENERJI_EK=_paux2.2 ENERJI_BUTCE_EK=_paux1.2 python faz5_enerji_erisim.py --kalibre 2>&1 | grep -E "ENERJIK +(NOM|K1|K13) " | head -3
echo "[$(zaman)] BITTI"
