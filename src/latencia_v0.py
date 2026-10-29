"""Latencia del v0: mediana y p90 en 15 fotos al azar por aparato (semilla 2026).

Uso:
    python src/latencia_v0.py --glucometro data/imagenes/glucometro --tensiometro data/imagenes/tensiometro
"""
import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import ocr_v0  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glucometro", required=True)
    ap.add_argument("--tensiometro", required=True)
    a = ap.parse_args()
    random.seed(2026)
    ts = []
    for d, tipo in [(a.glucometro, "glucometro"), (a.tensiometro, "tensiometro")]:
        fs = random.sample(sorted(f for f in os.listdir(d) if f.endswith(".jpg")), 15)
        for f in fs:
            t0 = time.time()
            ocr_v0.procesar(os.path.join(d, f), tipo)
            ts.append((tipo, time.time() - t0))
    for t in ["glucometro", "tensiometro"]:
        v = sorted(x for k, x in ts if k == t)
        print(t, "mediana", round(statistics.median(v), 1), "s · p90", round(v[int(0.9 * len(v)) - 1], 1), "s · n", len(v))


if __name__ == "__main__":
    main()
