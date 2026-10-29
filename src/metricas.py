"""Métricas de la E5 a partir de los CSV de predicciones (no necesita las fotos).

- Exactitud por foto: el valor leído es igual a la etiqueta (en el tensiómetro, los tres valores).
- Valor errado (contrapeso): la foto entrega al menos un valor y está mal.
- Rango: IC 95 % de Wilson; diferencia v0 − simple con bootstrap pareado (5 000 remuestreos, semilla 2026).
"""
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

SEMILLA = 2026
N_BOOT = 5000
CAMPOS_T = [("sistolica", "label_systole"), ("diastolica", "label_diastole"), ("pulso", "label_pulse")]


def igual(a, b):
    return (not pd.isna(a)) and (not pd.isna(b)) and int(round(float(a))) == int(round(float(b)))


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return round(100 * lo, 1), round(100 * hi, 1)


def cargar(data="data"):
    """Une las predicciones del modelo simple y del v0 por foto, con columnas ok_* y mal_*."""
    s = pd.read_csv(f"{data}/predicciones/simple.csv")
    g0 = pd.read_csv(f"{data}/predicciones/v0_glucometro.csv")
    t0 = pd.read_csv(f"{data}/predicciones/v0_tensiometro.csv")

    g = s[s.tipo == "glucometro"][["filename", "glucosa", "seg"]].rename(columns={"glucosa": "g_s"})
    g = g.merge(g0[["filename", "glucosa", "label_glucose", "ok_glucosa"]].rename(columns={"glucosa": "g_v0"}), on="filename")
    g["ok_s"] = [igual(a, b) for a, b in zip(g.g_s, g.label_glucose)]
    g["ok_v0"] = g.ok_glucosa.astype(bool)
    g["mal_s"] = g.g_s.notna() & ~g.ok_s
    g["mal_v0"] = g.g_v0.notna() & ~g.ok_v0

    t = s[s.tipo == "tensiometro"].merge(t0, on="filename", suffixes=("_s", ""))
    mal_s = np.zeros(len(t), bool)
    mal_v0 = np.zeros(len(t), bool)
    for c, l in CAMPOS_T:
        t[f"oks_{c}"] = [igual(a, b) for a, b in zip(t[c + "_s"], t[l])]
        t[f"ok_{c}"] = t[f"ok_{c}"].astype(bool)
        mal_s |= t[c + "_s"].notna().values & ~t[f"oks_{c}"].values
        mal_v0 |= t[c].notna().values & ~t[f"ok_{c}"].values
    t["ok_s"] = t[[f"oks_{c}" for c, _ in CAMPOS_T]].all(axis=1)
    t["ok_v0"] = t[[f"ok_{c}" for c, _ in CAMPOS_T]].all(axis=1)
    t["mal_s"], t["mal_v0"] = mal_s, mal_v0
    return g, t


def bootstrap_diferencia(d, rng):
    """Diferencia de exactitud v0 − simple en puntos, con IC 95 % por bootstrap pareado."""
    a, b = d.ok_v0.values.astype(int), d.ok_s.values.astype(int)
    idx = rng.integers(0, len(a), (N_BOOT, len(a)))
    diff = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return round(100 * (a.mean() - b.mean()), 1), round(100 * lo, 1), round(100 * hi, 1)


def tabla_baseline(g, t):
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for aparato, d in [("glucometro", g), ("tensiometro", t)]:
        n = len(d)
        for modelo, ok, mal in [("simple", "ok_s", "mal_s"), ("v0", "ok_v0", "mal_v0")]:
            k, m = int(d[ok].sum()), int(d[mal].sum())
            filas.append(dict(aparato=aparato, modelo=modelo, n=n, aciertos=k, exactitud=round(100 * k / n, 1),
                              ic95=wilson(k, n), errados=m, pct_errado=round(100 * m / n, 1), ic95_errado=wilson(m, n)))
        dif, lo, hi = bootstrap_diferencia(d, rng)
        filas.append(dict(aparato=aparato, modelo="v0 - simple (pp)", n=n, exactitud=dif, ic95=(lo, hi)))
    return pd.DataFrame(filas)


def segmentos_glucosa(g):
    g = g.copy()
    g["rango"] = pd.cut(g.label_glucose, [0, 99, 180, 1000], labels=["< 100", "100-180", "> 180"])
    r = g.groupby("rango", observed=True).ok_v0.agg(["sum", "count"]).rename(columns={"sum": "aciertos", "count": "n"})
    r["exactitud"] = (100 * r.aciertos / r.n).round(1)
    r["ic95"] = [wilson(k, n) for k, n in zip(r.aciertos, r.n)]
    return r
