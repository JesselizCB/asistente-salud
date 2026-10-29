# Asistente de salud por voz y cámara · Equipo 2

Capstone Project I · Maestría en Ciencia de Datos e IA · UTEC Posgrado (2026)
Christian Cordova · Meliza Sosa · Yesseliz Choque · Asesor: Boris Alzamora

Asistente en una tablet para adultos mayores con hipertensión y diabetes: lee por cámara la pantalla del glucómetro y del tensiómetro, repite el valor en voz alta y lo guarda solo si la persona lo confirma.

Este repositorio contiene la **evaluación del modelo v0 (OCR de pantallas)** de la E5.

## Ejecutar en 1 minuto (sin fotos)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/E5_evaluacion.ipynb   # o abrirlo y "Run all"
```

O en Colab: [notebooks/E5_evaluacion.ipynb](https://colab.research.google.com/github/JesselizCB/asistente-salud/blob/main/notebooks/E5_evaluacion.ipynb).

La última celda verifica que los números coinciden con la E5 (`OK: todos los números coinciden con la E5`).

| Modelo | Glucómetro (n = 120) | Tensiómetro (n = 120) |
|---|---|---|
| Modelo simple (OCR sin ajustes) | 55,0 % [46,1–63,6] | 2,5 % [0,9–7,1] |
| Modelo v0 (OCR + reglas) | 80,0 % [72,0–86,2] | 6,7 % [3,4–12,6] |
| Diferencia v0 − simple | +25,0 pp [15,8–34,2] | +4,2 pp [−1,7 a 10,0] |

Exactitud por foto con IC 95 % de Wilson; diferencia con bootstrap pareado (5 000 remuestreos).

## Semillas y decisiones fijas

- Semilla `2026` en el bootstrap (`src/metricas.py`) y en la muestra de latencia (`src/latencia_v0.py`).
- Conjunto de prueba congelado: los 120 + 120 archivos de `data/etiquetas/`.
- Acierto = el valor leído es igual a la etiqueta (en el tensiómetro, sistólica, diastólica y pulso).

## Estructura

```
src/ocr_v0.py            modelo v0: RapidOCR + 4 rotaciones, recorte LCD, votación, asignación por fila, filtro de rango
src/baseline_simple.py   modelo simple: RapidOCR sobre la foto completa
src/latencia_v0.py       latencia del v0 (mediana y p90)
src/metricas.py          exactitud, contrapeso, IC de Wilson y bootstrap pareado
data/etiquetas/          etiquetas de las 120 + 120 fotos y lista de la muestra ACC-1
data/predicciones/       salidas del v0 y del modelo simple (CSV)
notebooks/               notebook de la E5
```

## Volver a correr el OCR (necesita las fotos)

Copiar las fotos en `data/imagenes/glucometro/` y `data/imagenes/tensiometro/` (ver `data/imagenes/LEEME.md`) y luego:

```bash
python src/ocr_v0.py --tipo glucometro --imagenes data/imagenes/glucometro/ \
    --etiquetas data/etiquetas/glucometro_labels_120.csv --salida data/predicciones/v0_glucometro.csv
python src/ocr_v0.py --tipo tensiometro --imagenes data/imagenes/tensiometro/ \
    --etiquetas data/etiquetas/tensiometro_labels_120.csv --salida data/predicciones/v0_tensiometro.csv
python src/baseline_simple.py --glucometro data/imagenes/glucometro --tensiometro data/imagenes/tensiometro
python src/latencia_v0.py --glucometro data/imagenes/glucometro --tensiometro data/imagenes/tensiometro
```

## Datos y limitaciones

- **Fotos públicas de Kaggle (DataCluster Labs, 2021), no tomadas por el equipo.** Las fotos no se incluyen por peso (≈ 640 MB) y licencia.
- **Muestra ACC-1:** 100 fotos del mismo dataset (60 de glucómetro y 40 de tensiómetro) con la pantalla completa y enfocada, porque en el producto la guía por voz pide encuadrar antes de leer. Por ese criterio sus resultados pueden ser optimistas.
- Las reglas del v0 se ajustaron mirando estas fotos; ninguna etiqueta de glucosa es menor que 70 mg/dL.
- La validación externa será con fotos tomadas por el equipo al 14 de noviembre de 2026.
- Sin datos personales: solo fotos de pantallas, con nombres de archivo del dataset.
