#!/bin/bash
cd "$(dirname "$0")/.."
z() { date +%H:%M:%S; }
G="--gap 0.0005 --sure 1800"
echo "[$(z)] ana siki"; python faz11_optimizasyon.py --butce 100e6 $G --ek _siki 2>&1 | grep -E "Uc noktalar|eps|toplam|Error|Traceback"
echo "[$(z)] K12 siki"; python faz11_optimizasyon.py --butce 100e6 --k 12 $G --ek _K12siki 2>&1 | grep -E "Uc noktalar|eps|toplam|Error|Traceback"
echo "[$(z)] dc 60k"; python faz11_optimizasyon.py --butce 100e6 --dcfiyat 60000 $G --ek _dc60k 2>&1 | grep -E "Uc noktalar|eps|toplam|Error|Traceback"
echo "[$(z)] ac 12k"; python faz11_optimizasyon.py --butce 100e6 --acfiyat 12000 $G --ek _ac12k 2>&1 | grep -E "Uc noktalar|eps|toplam|Error|Traceback"
echo "[$(z)] sabit 0"; python faz11_optimizasyon.py --butce 100e6 --sabit 0 $G --ek _sabit0 2>&1 | grep -E "Uc noktalar|eps|toplam|Error|Traceback"
echo "[$(z)] nexp 2400"; python faz11_optimizasyon.py --butce 100e6 --nexp 2400 $G --ek _nexp2400 2>&1 | grep -E "Uc noktalar|eps|toplam|Error|Traceback"
echo "[$(z)] B25M siki"; python faz11_optimizasyon.py --butce 25e6 $G --ek _siki 2>&1 | grep -E "Uc noktalar|eps|toplam|Error|Traceback"
echo "[$(z)] pi5 siki"; python faz11_optimizasyon.py --butce 100e6 --pi 5 $G --ek _siki 2>&1 | grep -E "Uc noktalar|eps|toplam|Error|Traceback"
echo "[$(z)] MILP BITTI"
# ikinci revizyon (02.10.2026): yalniz baslangicta sifir olan dokuz ilceye esik (Bolum 4.5) ve asgari kapsama
echo "[$(z)] dokuz ilce hedefi"; python faz11_optimizasyon.py --butce 100e6 --gap 0.00001 --sure 1800 --hedef9 3.2062473739337582e-06
echo "[$(z)] asgari kapsama"; python faz11b_asgari_kapsama.py
echo "[$(z)] merkez paylari"; python faz11d_merkez_payi.py
