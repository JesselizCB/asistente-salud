"""Modelo simple (baseline): RapidOCR sobre la foto completa, sin orientar, recortar ni votar.

Glucómetro: el número más alto (en píxeles) dentro de rango.
Tensiómetro: los 3 números más grandes, ordenados de arriba abajo = sistólica, diastólica, pulso.

Uso:
    python src/baseline_simple.py --glucometro data/imagenes/glucometro \
        --tensiometro data/imagenes/tensiometro --salida data/predicciones/simple.csv
"""
import argparse
import csv
import os
import re
import time

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

E = RapidOCR()


def toks(path):
    img = cv2.imread(path)
    s = 1600 / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    res, _ = E(img)
    out = []
    for caja, t, c in res or []:
        v = re.sub(r"[^\d]", "", t)
        if re.fullmatch(r"\d{2,3}", v) and not re.search(r"[/:]", t):
            b = np.array(caja, float)
            out.append((int(v), np.ptp(b[:, 1]), b[:, 1].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glucometro", required=True)
    ap.add_argument("--tensiometro", required=True)
    ap.add_argument("--salida", default="data/predicciones/simple.csv")
    a = ap.parse_args()
    rows = []
    for tipo, d in [("glucometro", a.glucometro), ("tensiometro", a.tensiometro)]:
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".jpg"):
                continue
            t0 = time.time()
            tk = toks(os.path.join(d, f))
            dt = time.time() - t0
            tk.sort(key=lambda x: -x[1])
            if tipo == "glucometro":
                g = next((v for v, h, y in tk if 20 <= v <= 600), None)
                rows.append(dict(tipo=tipo, filename=f, glucosa=g, sistolica=None, diastolica=None, pulso=None, seg=round(dt, 2)))
            else:
                top = sorted(tk[:3], key=lambda x: x[2]) + [(None, 0, 0)] * 3
                rows.append(dict(tipo=tipo, filename=f, glucosa=None, sistolica=top[0][0], diastolica=top[1][0], pulso=top[2][0], seg=round(dt, 2)))
    with open(a.salida, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "fotos ->", a.salida)


if __name__ == "__main__":
    main()
