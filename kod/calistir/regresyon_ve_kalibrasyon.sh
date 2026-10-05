#!/bin/bash
# Ilce regresyonu (Tablo 6) ve enerji butcesi kalibrasyonunun ornek oynakligi (Bolum 3.3.2)
cd "$(dirname "$0")/.."
python faz9c_boyutsuz.py
for t in 1 2 3; do python faz5_kalibre.py --tohum $t | grep ESLESEN; done
