# WiDS Worldwide Datathon 2026 — Plantilla de equipo

> **Reto:** *Predicting Time-to-Threat for Evacuation Zones Using Survival Analysis.*
> Estimar, dentro de las próximas 48 h, la probabilidad de que un incendio activo amenace zonas
> de evacuación / infraestructura crítica (líneas de transmisión, vías, servicios públicos, corredores críticos).

**Aliado del reto:** [Watch Duty](https://www.watchduty.org/) — organización sin fines de lucro que entrega
alertas hiperlocales en tiempo real durante incendios forestales.

Este repositorio es la **plantilla base** de cada equipo. Trabajen sobre él, hagan commits frecuentes y
entreguen el **link del repo + el commit hash final** antes del cierre.

---

## 👥 Integrantes del equipo

| Nombre | GitHub | Rol (opcional) |
|--------|--------|----------------|
| _Michael Estrada_ | @BryanEstrada003 |  |
| _Keyla Franco_ | @KeylaFrancoH | |
| _Jonathan García_ | @ElPitagoras14| |
| _Alexa Marin_ | @aamarin05| |

**Nombre del equipo:** _Data Wizards_

---

## 🔥 El problema

Se trata de un problema de **análisis de supervivencia (survival analysis)**. Para cada evento de incendio
se observan sus **primeras 5 horas** y se quiere predecir cuándo (y si) el fuego llega a ≤ 5 km de una zona
de evacuación dentro de una ventana de 72 h.

- `time_to_hit_hours` — tiempo (horas) desde t0+5h hasta que el fuego llega a ≤ 5 km. Para casos **censurados**
  (nunca llegan dentro de 72 h) es el último tiempo observado (≤ 72).
- `event` — indicador: `1` si el fuego llegó dentro de 72 h, `0` si está censurado.

La **submission** no se hace sobre esas columnas crudas, sino sobre **probabilidades a varios horizontes**:
`prob_12h, prob_24h, prob_48h, prob_72h` (ver `sample_submission.csv`).

El diccionario completo de las 37 columnas está en `metaData.csv` y, en versión estática, dentro del
notebook guía (`notebooks/Notebook_starter_WIDS_2026.ipynb`).

---

## 📁 Estructura del repositorio

```
.
├── README.md                     # este archivo
├── pyproject.toml                # dependencias
├── .gitignore                    # excluye datos pesados y artefactos
├── notebooks/
│   └── Notebook_starter_WIDS_2026.ipynb   # notebook guía oficial (punto de partida)
├── src/                          # scripts modulares (features, modelos, utils)
│   └── README.md
└── submissions/                  # cada submission con timestamp
    └── README.md
```

---

## 🚀 Cómo correr el código

### Opción A — Google Colab (recomendado, sin instalar nada)
1. Sube `notebooks/Notebook_starter_WIDS_2026.ipynb` a [Colab](https://colab.research.google.com/).
2. Ejecuta las celdas en orden. La primera celda instala dependencias y la siguiente descarga los datos
   automáticamente.

### Opción B — Local
```bash
# 1. Clonar
git clone <URL_DE_ESTE_REPO>.git
cd <repo>

# 2. Entorno
python -m venv .venv && source .venv/bin/activate   # venv
pip install .

# 3. Abrir el notebook
jupyter lab notebooks/Notebook_starter_WIDS_2026.ipynb
```

Los datos se descargan dentro del notebook desde el repositorio oficial de datos del datathon; **no** se
versionan en este repo (ver `.gitignore`).

---

## 📤 Entregables (recordatorio)

| # | Entregable | Formato | Obligatorio |
|---|-----------|---------|:---:|
| 1 | Predicciones sobre el test | `submission.csv` con el formato exacto de `sample_submission.csv` | ✅ |
| 2 | Notebook / código fuente reproducible | `.ipynb` o este repositorio | ✅ |
| 3 | Reporte ejecutivo (insights + metodología) | PDF o slides, máx. 10 págs | ✅ |
| 4 | Datasets externos usados (si aplica) | Lista referenciada en el reporte | ❌ |

**Reglas del repo:**
- Entregar el **link del repositorio** y el **commit hash final** antes del cierre.
- Los repos se hacen **públicos** al cerrar el evento.
- Guardar cada submission en `submissions/` con un timestamp / nombre de experimento.

---

## 🔗 Link de la presentacion

Link de Reporte Ejecutivo: [Reporte Ejecutivo](insertar link aqui)

---

¡Éxitos! 🔥

