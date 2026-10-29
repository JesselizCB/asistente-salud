"""
CuidaFamilia IA · OCR de pantallas de tensiómetro y glucómetro
Capstone E4 · Equipo 02 · evidencia para H4 (glucómetro) y H5 (tensiómetro)

Motor: RapidOCR (modelos PP-OCR de PaddleOCR exportados a ONNX), libre y corre en CPU.
    pip install rapidocr_onnxruntime opencv-python pandas scipy

Uso:
    # Tensiómetro (H5)
    python src/ocr_v0.py --tipo tensiometro --imagenes data/imagenes/tensiometro/ \
        --etiquetas data/etiquetas/tensiometro_labels_120.csv --salida data/predicciones/v0_tensiometro.csv

    # Glucómetro (H4)
    python src/ocr_v0.py --tipo glucometro --imagenes data/imagenes/glucometro/ \
        --etiquetas data/etiquetas/glucometro_labels_120.csv --salida data/predicciones/v0_glucometro.csv

    # Muestra fija de 120 imágenes (semilla registrada en el notebook)
    python src/ocr_v0.py ... --muestra 120 --semilla 2026

Pipeline por imagen:
    1. Orientación: prueba 0/90/180/270° y se queda con la que lee mejor el texto
       impreso del equipo (SYS/DIA/PUL, marca).
    2. Recorte de la pantalla LCD (contorno rectangular más cercano a las etiquetas).
    3. OCR en 4 variantes (2 escalas × {gris, segmentos unidos}) y votación.
       La erosión une los segmentos de los dígitos de 7 segmentos, que el OCR
       genérico suele leer partidos.
    4. Asignación por POSICIÓN: cada número va a la fila SYS/DIA/PUL más cercana.
       Es la misma convención de las etiquetas del dataset (un valor en la fila
       DIA se etiqueta como diastólica aunque la pantalla muestre solo ese valor).
    5. Validación fisiológica: lo que cae fuera de rango se descarta y se marca.
"""

import argparse
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from scipy.stats import beta

ENGINE = RapidOCR()
ROTACIONES = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
ANCLAS = {"SYS": ("SYS", "SYS", "SXS"), "DIA": ("DIA", "PIA"), "PUL": ("PUL",)}
RANGOS = {"sistolica": (60, 260), "diastolica": (30, 150), "pulso": (30, 200), "glucosa": (20, 600)}


# ---------------------------------------------------------------- utilidades
def cargar(ruta, lado_max=1600):
    img = cv2.imread(ruta)
    if img is None:
        raise ValueError(f"No se pudo leer {ruta}")
    s = lado_max / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def ocr(img, cls=True):
    res, _ = ENGINE(img, use_cls=cls)
    out = []
    for caja, texto, conf in res or []:
        b = np.array(caja, dtype=float)
        out.append({"texto": texto, "conf": float(conf),
                    "cx": b[:, 0].mean(), "cy": b[:, 1].mean(), "h": np.ptp(b[:, 1])})
    return out


def buscar_anclas(tokens):
    anclas = {}
    for t in tokens:
        txt = t["texto"].upper()
        for k, variantes in ANCLAS.items():
            if k not in anclas and any(v in txt for v in variantes):
                anclas[k] = t["cy"]
    # Si faltan una o dos, se infieren asumiendo filas equidistantes
    if len(anclas) == 2:
        orden = ["SYS", "DIA", "PUL"]
        i = sorted(orden.index(k) for k in anclas)
        y = [anclas[orden[j]] for j in i]
        paso = (y[1] - y[0]) / (i[1] - i[0])
        falta = [k for k in orden if k not in anclas][0]
        anclas[falta] = y[0] + (orden.index(falta) - i[0]) * paso
    return anclas


# ---------------------------------------------------------------- 1. orientación
def orientar(img):
    mejor = None
    for r in ROTACIONES:
        im = img if r is None else cv2.rotate(img, r)
        toks = ocr(im)
        anclas = {k: v for k, v in buscar_anclas(toks).items()}
        texto_ok = sum(t["conf"] for t in toks if len(re.sub("[^A-Za-z]", "", t["texto"])) >= 3)
        puntaje = 3 * len(anclas) + texto_ok
        if mejor is None or puntaje > mejor[0]:
            mejor = (puntaje, im, anclas)
    return mejor[1], mejor[2]


# ---------------------------------------------------------------- 2. pantalla LCD
def recortar_lcd(im, anclas):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    bordes = cv2.dilate(cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 30, 90), None, iterations=2)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if 0.02 * H * W < w * h < 0.5 * H * W and 0.5 < w / h < 2.2 and cv2.contourArea(c) > 0.6 * w * h:
            cands.append((x, y, w, h))
    if not cands:
        return im
    if anclas:
        ay = np.mean(list(anclas.values()))
        cands.sort(key=lambda b: abs(b[1] + b[3] / 2 - ay) - 0.0005 * b[2] * b[3])
    else:
        cands.sort(key=lambda b: -b[2] * b[3])
    x, y, w, h = cands[0]
    return im[y:y + h, x:x + w]


# ---------------------------------------------------------------- 3. OCR con votación
def variantes(lcd):
    g0 = cv2.cvtColor(lcd, cv2.COLOR_BGR2GRAY)
    for lado in (700, 1100, 1500):
        s = lado / max(g0.shape)
        g = cv2.resize(g0, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        g = cv2.createCLAHE(2.5, (8, 8)).apply(g)
        k = int(max(3, g.shape[0] / 90)) | 1
        unido = cv2.erode(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        for v in (g, unido):
            yield cv2.copyMakeBorder(v, 40, 40, 40, 40, cv2.BORDER_REPLICATE), s, 40


def leer_numeros(lcd):
    """Devuelve números candidatos (en coordenadas del recorte) y anclas del recorte."""
    candidatos, anclas_lcd = [], {}
    for img, s, pad in variantes(lcd):
        toks = ocr(img, cls=False)  # sin clasificador de ángulo: voltea dígitos LCD
        for t in toks:
            t["cx"], t["cy"], t["h"] = (t["cx"] - pad) / s, (t["cy"] - pad) / s, t["h"] / s
        for k, v in buscar_anclas(toks).items():
            anclas_lcd.setdefault(k, []).append(v)
        for t in toks:
            txt = t["texto"]
            if re.search(r"[/:]", txt):          # fecha u hora
                continue
            limpio = re.sub(r"[^\d.]", "", txt)
            if not re.fullmatch(r"\d{1,3}(\.\d)?", limpio):
                continue
            t["valor"] = limpio
            candidatos.append(t)
    # Votación: agrupa lecturas del mismo lugar y elige la más repetida
    grupos = []
    for t in sorted(candidatos, key=lambda t: -t["h"]):
        for gr in grupos:
            if abs(gr[0]["cy"] - t["cy"]) < 0.5 * gr[0]["h"] and abs(gr[0]["cx"] - t["cx"]) < gr[0]["h"]:
                gr.append(t)
                break
        else:
            grupos.append([t])
    numeros = []
    for gr in grupos:
        # Puntaje = suma de confianzas; en empate gana el más largo (el OCR suele perder un "1")
        puntaje = pd.Series([t["conf"] for t in gr]).groupby([t["valor"] for t in gr]).sum()
        valor = max(puntaje.index, key=lambda v: (round(puntaje[v], 1), len(v)))
        numeros.append({"valor": valor, "cy": np.mean([t["cy"] for t in gr]),
                        "h": max(t["h"] for t in gr), "votos": len(gr)})
    anclas_lcd = {k: float(np.median(v)) for k, v in anclas_lcd.items()}
    return numeros, anclas_lcd


# ---------------------------------------------------------------- 4–5. asignación y validación
def en_rango(campo, v):
    lo, hi = RANGOS[campo]
    return v is not None and lo <= v <= hi


def asignar_tensiometro(numeros, anclas):
    pred = {"sistolica": None, "diastolica": None, "pulso": None}
    banderas = []
    if not numeros:
        return pred, ["sin_lectura"]
    hmax = max(n["h"] for n in numeros)
    numeros = [n for n in numeros if n["h"] >= 0.25 * hmax and "." not in n["valor"]]
    mapa = {"SYS": "sistolica", "DIA": "diastolica", "PUL": "pulso"}
    if len(anclas) >= 2:
        # Las etiquetas impresas (SYS/DIA/PUL) quedan en el borde superior de su fila:
        # cada número pertenece a la última etiqueta que está por encima de él.
        ys = sorted(anclas.values())
        paso = np.median(np.diff(ys)) if len(ys) > 1 else hmax
        tol = 0.3 * paso
        for n in sorted(numeros, key=lambda n: -n["h"]):
            if n["cy"] < min(ys) - tol:            # reloj / fecha / n.º de memoria
                continue
            arriba = {k: y for k, y in anclas.items() if y <= n["cy"] + tol}
            fila = max(arriba, key=arriba.get) if arriba else min(anclas, key=lambda k: abs(anclas[k] - n["cy"]))
            campo = mapa[fila]
            if campo != "pulso" and n["h"] < 0.5 * hmax:   # dígitos chicos fuera de PUL = fecha
                continue
            if pred[campo] is None:
                pred[campo] = int(n["valor"])
    else:
        banderas.append("sin_anclas")
        grandes = sorted([n for n in numeros if n["h"] >= 0.6 * hmax], key=lambda n: n["cy"])
        chicos = [n for n in numeros if n["h"] < 0.6 * hmax and n["cy"] > (grandes[-1]["cy"] if grandes else 0)]
        for campo, n in zip(["sistolica", "diastolica"], grandes):
            pred[campo] = int(n["valor"])
        if chicos:
            pred["pulso"] = int(max(chicos, key=lambda n: n["h"])["valor"])
    for campo, v in pred.items():
        if v is not None and not en_rango(campo, v):
            banderas.append(f"{campo}_fuera_de_rango({v})")
            pred[campo] = None
    if pred["sistolica"] and pred["diastolica"] and pred["sistolica"] <= pred["diastolica"]:
        banderas.append("PAS<=PAD")
    return pred, banderas


# De todas las imagenes, ninguna tiene unidad mmol/L, todas están en mg/dL
# actualicemos esta función para reflejar que todas las imágenes están en mg/dL
# además quitemos fuera de rango
def asignar_glucometro(numeros):
    if not numeros:
        return {"glucosa": None, "unidad": None}, ["sin_lectura"]
    n = max(numeros, key=lambda n: n["h"])     # el valor de glucosa es el dígito más grande
    glucosa_value = int(n["valor"])
    banderas = []
    return {"glucosa": glucosa_value, "unidad": "mg/dL"}, banderas


def procesar(ruta, tipo):
    img = cargar(ruta)
    im, anclas = orientar(img)
    lcd = recortar_lcd(im, anclas)
    numeros, anclas_lcd = leer_numeros(lcd)
    if tipo == "tensiometro":
        pred, flags = asignar_tensiometro(numeros, anclas_lcd)
    else:
        pred, flags = asignar_glucometro(numeros)
    return {"filename": os.path.basename(ruta), **pred,
            "leidos": " ".join(n["valor"] for n in numeros), "banderas": ";".join(flags)}


# ---------------------------------------------------------------- evaluación
def clopper_pearson(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def igual(p, e):
    if pd.isna(e):
        return pd.isna(p)
    return not pd.isna(p) and int(round(float(p))) == int(round(float(e)))


def evaluar(pred, etiquetas, tipo, n_minimo):
    df = pred.merge(etiquetas, on="filename", how="inner")
    if tipo == "tensiometro":
        campos = [("sistolica", "label_systole"), ("diastolica", "label_diastole"), ("pulso", "label_pulse")]
        for c, l in campos:
            df[f"ok_{c}"] = [igual(p, e) for p, e in zip(df[c], df[l])]
            print(f"  {c:<11} acierto por campo: {df[f'ok_{c}'].mean():.1%}")
        # Criterio pre-registrado H5: imágenes con los 3 valores etiquetados y los 3 correctos
        base = df.dropna(subset=[l for _, l in campos])
        base = base[base.label_systole.between(60, 260) & base.label_diastole.between(30, 150)
                    & base.label_pulse.between(30, 200)]
        k = int(base[[f"ok_{c}" for c, _ in campos]].all(axis=1).sum())
    else:
        df["ok_glucosa"] = [igual(p, e) for p, e in zip(df["glucosa"], df["label_glucose"])]
        base = df.dropna(subset=["label_glucose"])
        k = int(base["ok_glucosa"].sum())
    n = len(base)
    if n == 0:
        print("  No hay imágenes evaluables.")
        return df
    lo, hi = clopper_pearson(k, n)
    print(f"  Acierto (criterio pre-registrado): {k}/{n} = {k / n:.1%}  IC95% [{lo:.1%} – {hi:.1%}]")
    if n < n_minimo:
        print(f"  ATENCIÓN: n = {n} < {n_minimo} → según el pre-registro la hipótesis es NO TESTEABLE.")
    return df


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", choices=["tensiometro", "glucometro"], required=True)
    ap.add_argument("--imagenes", required=True)
    ap.add_argument("--etiquetas")
    ap.add_argument("--salida", default="predicciones.csv")
    ap.add_argument("--muestra", type=int)
    ap.add_argument("--semilla", type=int, default=2026)
    a = ap.parse_args()

    rutas = sorted(glob.glob(os.path.join(a.imagenes, "*.jp*g")))
    if a.muestra and a.muestra < len(rutas):
        rutas = list(pd.Series(rutas).sample(a.muestra, random_state=a.semilla))
    filas = []
    for i, r in enumerate(rutas, 1):
        try:
            filas.append(procesar(r, a.tipo))
        except Exception as e:  # una imagen corrupta no detiene el lote
            filas.append({"filename": os.path.basename(r), "banderas": f"error:{e}"})
        print(f"[{i}/{len(rutas)}] {filas[-1]}")
    pred = pd.DataFrame(filas)

    if a.etiquetas:
        print("\nEvaluación contra etiquetas")
        pred = evaluar(pred, pd.read_csv(a.etiquetas), a.tipo,
                       n_minimo=106 if a.tipo == "tensiometro" else 100)
    pred.to_csv(a.salida, index=False)
    print(f"\nPredicciones guardadas en {a.salida}")


if __name__ == "__main__":
    main()
