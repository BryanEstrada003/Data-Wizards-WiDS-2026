# Decisiones — Defensa técnica

> WiDS Datathon 2026 · Data Wizards — *Predicting Time-to-Threat for Evacuation Zones*
> Documento de respaldo para defender las decisiones metodológicas del proyecto.

## Contexto mínimo (necesario para entender el resto)

Problema de **análisis de supervivencia con censura a derecha**: de cada incendio se observan sus
primeras 5 h y se predice la probabilidad de que llegue a ≤ 5 km de una zona de evacuación a los
horizontes 12 / 24 / 48 / 72 h. `event=1` si llegó dentro de 72 h, `event=0` si censurado.

- **221** observaciones en train (**69 eventos = 31.2 %**, 152 censurados), **95** en test.
- **Hallazgo que condiciona todo:** hay **separación perfecta a 5 km** — los 69 `event=1` están todos
  < 5 km y los 152 `event=0` todos ≥ 5 km. La variable `dist_min_ci_0_5h` domina cualquier modelo
  (log-rank cerca/lejos: **p ≈ 1.5 × 10⁻⁷⁰**).
- En test hay una **zona gris (5–15 km, 7 incendios)** que el train nunca observó.

---

## 📊 Resumen — Hiperparámetros y escenarios probados (con scores)

> Vista de un vistazo: qué hiperparámetros se eligieron, qué rango se exploró y el score de cada
> escenario. El detalle metodológico de cada decisión está en las secciones §1–§7.

### Hiperparámetros por modelo (elegido + espacio de búsqueda)

| Modelo | Hiperparámetro | Valores / rango probado | **Elegido** |
|--------|----------------|-------------------------|:-----------:|
| **RSF** (`scikit-survival`) | `n_estimators` | 300, 500 | **300** |
| | `min_samples_leaf` | 5, 10, 15, 20 | **10** |
| | `max_features` | `"sqrt"`, 0.5 | **0.5** |
| | `random_state` / `n_jobs` | 42 / −1 (fijos) | — |
| **Cox PH** (`lifelines`) | `penalizer` (L2) | 0.0, 0.05, 0.10 | **0.05** |
| **KM estratificado** | estratificación | cuartiles de `log(dist_min_ci_0_5h)` | **4 cuartiles** |

El grid del RSF **no es factorial completo** (2×4×2 = 16): se evaluó un **conjunto curado de 6
combinaciones**, deliberadamente pequeño para n=221 y evitar overfitting (`model_rsf.py:200`):

```
(300, 5,  sqrt)   (300, 10, sqrt)   (300, 15, sqrt)
(300, 10, 0.5) ←  (500, 10, sqrt)   (300, 20, sqrt)
   mejor
```

El `penalizer` del Cox se fijó en **0.05** como L2 para estabilizar con n=221 (`model_cox.py:57`); el EDA
univariado exploratorio usó 0.1 y el ablation sin regularización 0.0.

### Escenarios probados, con scores (CV 5-fold estratificado por evento)

| Modelo | Escenario | n_feat | C-index CV | IBS | Estado |
|--------|-----------|:------:|:----------:|:---:|--------|
| KM estratificado | cuartiles `log_dist` | — | — | — | Referencia (solo 4 valores/horizonte) |
| **RSF** | Todas las features | 33 | 0.9485 ± 0.0061 | — | — |
| **RSF** | **Seleccionadas (final)** | 24 | **0.9481 ± 0.0049** | **0.0304** | ✅ Submission generado |
| RSF | Aumentado (+10 nuevas) | 34 | 0.9517 ± 0.0080 | 0.0307 | Descartado (ΔC en el ruido, IBS peor) |
| Cox (lifelines) | Todas las features | 33 | 0.8655 ± **0.0986** | — | Descartado (varianza por colinealidad) |
| **Cox (lifelines)** | **Seleccionadas VIF ≤ 5 (final)** | 11 | **0.9219 ± 0.0228** | **0.0654 ± 0.0259** | ✅ Modelo activo |
| Cox (lifelines) | + ratio/escala | 15 | 0.9270 ± 0.0048 | (IBS +0.005) | Descartado (ninguna significativa) |
| Cox (lifelines) | + cinemáticas | 12 | 0.9256 ± 0.0027 | — | Descartado (ΔC +0.0003) |

**Brier Score por horizonte del Cox final** (lifelines, 11 feat): BS@12h = 0.0744 ± 0.0202 ·
BS@24h = 0.0664 ± 0.0268 · BS@48h = 0.0593 ± 0.0310. Referencia del modelo nulo (predecir 0.5): IBS = 0.25
→ el Cox (0.065) es **~4× mejor**.

**Experimento aparte — efecto de los logs** (sobre la data **original cruda** con L2 leve, para aislar el
efecto de la colinealidad; `feature_selection.ipynb` §2.5, fig 22):

| Escenario Cox (data original) | n_feat | C-index | C_std | IBS |
|-------------------------------|:------:|:-------:|:-----:|:---:|
| Original (todas, colineales) | 34 | 0.816 | 0.086 | 0.145 |
| **Con logs (1 representante, −14 colineales)** | 20 | **0.839** | **0.050** | 0.151 |
| Sin logs (mantiene colineales) | 31 | 0.810 | 0.092 | 0.144 |

> En ese mismo experimento el **RSF se mantiene invariante (C ≈ 0.948 en los tres escenarios)**: la poda de
> colineales solo beneficia al Cox lineal (§5.1–§5.2).

**Modelo seleccionado:** RSF (24 features) para el submission por su C-index (0.948); Cox (11 features,
lifelines) como modelo interpretable y mejor reportado en calibración (IBS 0.065). KM como referencia.

---

## 1. Por qué análisis de supervivencia y no clasificación binaria

**Decisión:** modelar el problema como **time-to-event (supervivencia)** y no como una clasificación
binaria independiente en cada horizonte.

### Justificación

1. **El 68.8 % de los datos están censurados.** Un censurado (`event=0`) no significa "no llega nunca",
   significa "no llegó *dentro de la ventana observada* (≤ 72 h)". Un clasificador binario obligaría a
   tratar ese caso como un negativo definitivo, lo cual es **falso** para los censurados que podrían
   haber llegado a las 80 h. La censura es información parcial, no una etiqueta negativa.

2. **El target tiene dos componentes acoplados** (`time_to_hit_hours` + `event`) que un solo número
   binario no captura. Supervivencia modela la función S(t) completa y de ahí derivamos los 4 horizontes
   de forma coherente, en lugar de entrenar 4 clasificadores sueltos que podrían contradecirse entre sí
   (p. ej. predecir mayor prob. a 24 h que a 48 h — ver decisión de monotonicidad en §3).

3. **La submission pide probabilidades a múltiples horizontes** (`prob_12h…72h`). Una única S(t) por
   incendio produce los 4 valores de manera consistente y monótona; 4 modelos binarios no garantizan esa
   coherencia.

### Por qué no las alternativas

| Alternativa | Razón de descarte |
|-------------|-------------------|
| **Clasificación binaria** (1 modelo por horizonte) | Descarta o sesga la censura; no garantiza monotonicidad entre horizontes; multiplica el riesgo de sobreajuste con n=221. |
| **Regresión sobre `time_to_hit_hours`** | Ignora `event`: trataría los tiempos censurados como tiempos reales de evento, sesgando todo (mismo problema que la correlación con el tiempo, §4). |

### Preguntas probables

- *¿Por qué no un clasificador por horizonte, más simple?* → Porque trataría a 152 censurados como
  negativos definitivos, cuando solo sabemos que no llegaron en la ventana observada. Supervivencia usa
  esa información parcial correctamente.
- *¿Qué gano con S(t)?* → Una sola curva por incendio que da los 4 horizontes de forma coherente y
  monótona, en vez de 4 predicciones que podrían contradecirse.

---

## 2. Por qué C-index + IBS/Brier (IPCW) como métricas, y no accuracy/AUC

**Decisión:** evaluar con **C-index** (discriminación / orden de riesgo) e **Integrated Brier Score y
Brier Score por horizonte con IPCW** (calibración de las probabilidades).

### Justificación

1. **Accuracy no aplica con censura.** Para calcular accuracy necesitaríamos una etiqueta verdadera en
   cada horizonte, y para los censurados esa etiqueta no existe. El **C-index** es la generalización
   natural del AUC a datos censurados: mide la probabilidad de **ordenar bien** dos incendios por su
   riesgo (0.5 = azar, 1.0 = perfecto), usando solo los pares comparables (donde la censura lo permite).

2. **El C-index mide orden, no calibración.** Pero la submission se evalúa sobre **probabilidades**
   (`prob_Xh`), no sobre un ranking. Por eso añadimos el **Brier Score** (error cuadrático entre
   probabilidad predicha y resultado observado) y su versión integrada **IBS**. El ajuste **IPCW**
   (Inverse Probability of Censoring Weighting) corrige el sesgo que la censura introduce en el Brier.

3. **Referencia interpretable:** el IBS del modelo nulo (predecir 0.5 siempre) es 0.25. Nuestro Cox
   logra **IBS ≈ 0.065**, ~4× mejor — una métrica que sí se puede comunicar a un jurado.

En resumen: **C-index responde "¿ordena bien el riesgo?"** y **Brier/IBS responde "¿son creíbles las
probabilidades?"**. Necesitamos ambas porque la competencia puntúa probabilidades.

### Preguntas probables

- *¿Por qué no AUC?* → El C-index *es* la versión de AUC para supervivencia con censura; AUC clásico
  necesitaría etiquetas binarias que los censurados no tienen.
- *¿Por qué dos métricas?* → Un modelo puede ordenar bien (C-index alto) pero estar mal calibrado
  (Brier malo). Como nos puntúan probabilidades, vigilamos las dos.
- *¿Qué es IPCW y por qué?* → Pondera para corregir que los censurados "desaparecen" antes de tiempo;
  sin ese ajuste el Brier estaría sesgado.

---

## 3. Estrategia de validación (CV estratificado + nested vs global + adversarial)

**Decisión:** **CV 5-fold estratificado por `event`**, con **validación anidada (nested) para la
selección de features** y **adversarial validation** para comprobar que el CV en train es confiable
para test.

### 3.1 Por qué 5-fold estratificado por evento

Con n=221 y solo 69 eventos, un fold mal balanceado podría quedarse con muy pocos eventos y dar una
métrica ruidosa o engañosa. Estratificar por `event` mantiene la proporción ~31 % de eventos en cada
fold. 5-fold es el compromiso estándar entre sesgo y varianza para este tamaño de muestra (cada fold de
validación tiene ~44 incendios, ~14 eventos).

### 3.2 Nested feature validation vs selección global — **por qué importa**

Este es el punto crítico de la validación, y la razón de la figura 29.

- **Enfoque "flat" / global (incorrecto para estimar desempeño):** seleccionar las features **una sola
  vez usando todo el train** y luego hacer CV con ese conjunto ya fijado. El problema: los folds de
  validación **ya participaron** en decidir qué features entran. Hay **fuga de información (leakage)**:
  el modelo "vio" los datos de validación durante la selección, así que el C-index resultante es
  **optimista** — más alto de lo que será en test.

- **Enfoque "nested" / anidado (correcto):** la selección de features se **rehace dentro de cada fold**,
  usando solo los datos de entrenamiento de ese fold. El fold de validación nunca influye en qué
  features se eligen. El C-index resultante es una **estimación honesta** de lo que veremos en datos
  nuevos.

- **El "optimismo" es la diferencia flat − nested.** Cuantifica exactamente cuánto inflamos la métrica
  al seleccionar features mirando todos los datos. Si esa brecha es grande, nuestro "0.948" sería un
  espejismo; si es pequeña, la selección es estable y la métrica es confiable.

> **Por qué es especialmente importante aquí:** hicimos selección de features agresiva (33 → 11 en Cox,
> 33 → 24 en RSF). Cualquier selección basada en el target es una vía de leakage; **reportar solo la CV
> flat sobre features ya seleccionadas sería auto-engañarse.** El nested CV es el que nos da derecho a
> defender el C-index. En nuestro caso la brecha de optimismo resultó pequeña (consistente con el bajo
> drift train/test), lo que confirma que el desempeño no está inflado por la selección. Ver figura 29.

### 3.3 Adversarial validation (¿el CV en train sirve para test?)

Se entrena un clasificador para distinguir filas de train vs test. Si el **AUC ≈ 0.5**, train y test son
indistinguibles → el CV en train es un buen proxy del test. Resultado: AUC cercano a 0.5 y **drift KS
máximo ~0.11** (no significativo) → la validación es transferible al test.

### Preguntas probables

- *¿Cómo sé que 0.948 no es optimista?* → Por el nested CV: rehacemos la selección de features dentro de
  cada fold, así el fold de validación nunca influye en qué features entran. La brecha flat−nested
  (optimismo) resultó pequeña.
- *¿No basta con CV normal?* → No, si seleccionas features con todo el train antes de la CV, hay
  leakage. La CV normal sobre features ya elegidas da un número inflado.
- *¿El CV en train predice el test?* → Sí: la adversarial validation da AUC ~0.5 y el drift KS máx es
  ~0.11, así que train y test son estadísticamente similares.

---

## 4. Por qué Spearman y no otra correlación

**Decisión:** usar correlación de **Spearman** (rangos) para el heatmap entre features (`eda.py:254`)
y para la correlación feature–target (`eda.py:277-289`).

### Justificación

Spearman es Pearson aplicado **sobre los rangos** en vez de sobre los valores crudos. Eso resuelve tres
problemas concretos de estos datos:

1. **Las relaciones son monótonas pero no lineales.** El riesgo cae con la distancia de forma
   logarítmica (por eso usamos `log_dist_min`, `log1p_area_first`, `log1p_growth`). Pearson solo mide
   relación lineal y subestimaría estas asociaciones; Spearman captura cualquier relación monótona.

2. **Hay outliers y colas largas.** El EDA detecta features con `|z| > 3` (sección 8) y
   `dist_min_ci_0_5h` abarca de ~100 m a ~500 km (3 órdenes de magnitud). Pearson se deja arrastrar por
   los extremos; Spearman, al trabajar con rangos, es robusto: el incendio más lejano es "el rango N"
   sin importar si está a 300 o 3000 km.

3. **Invariancia a las transformaciones monótonas que ya aplicamos.** `area_first_ha` y
   `log1p_area_first` tienen el mismo orden, así que su Spearman con el target es idéntico. No hace
   falta decidir si transformar *antes* de medir la correlación.

### Por qué no las alternativas

| Método | Razón de descarte |
|--------|-------------------|
| **Pearson** | Asume linealidad y lo distorsionan los outliers/sesgo que el propio EDA documenta. |
| **Kendall's τ** | También basado en rangos y algo más robusto con n pequeño/empates, pero más lento y con conclusiones casi idénticas. Spearman es el estándar convencional. |
| **Mutual information / distance correlation** | Capturan relaciones no monótonas, pero no dan **signo** (¿protector o riesgo?) ni se interpretan bien en un heatmap. Pérdida de interpretabilidad sin ganancia aquí. |

### Limitación reconocida (importante para la defensa)

La correlación de cada feature con **`time_to_hit_hours`** (`eda.py:284-289`) trata los tiempos
censurados como si fueran tiempos de evento reales, y el 68.8 % de los datos están censurados → esa
correlación está **sesgada**. No es un error: la figura 04 es solo exploratoria para *priorizar*
features. El análisis válido del target censurado se hace después con **Kaplan-Meier + log-rank**
(sección 6) y **Cox** (sección 11), que sí manejan censura correctamente.

### Preguntas probables

- *¿Por qué no Pearson, que es lo más común?* → Porque asume linealidad y es sensible a outliers, y
  nuestros datos son sesgados, con colas largas y relaciones logarítmicas. Spearman es robusto a las tres.
- *¿La correlación con el tiempo no ignora la censura?* → Sí, y lo sabemos; esa vista es solo para
  ranking exploratorio. La inferencia real sobre el tiempo la hacen KM/log-rank y Cox.
- *¿Spearman no pierde información al usar rangos?* → Pierde la magnitud exacta, pero gana robustez; y
  como solo lo usamos para detectar redundancia y priorizar, la magnitud lineal no aporta.

---

## 5. Por qué se descartaron variables y cómo se validó

**Decisión:** reducir el conjunto de features con **dos criterios independientes según el modelo**, y
validar cada descarte para no perder señal.

### 5.1 Cox PH — VIF iterativo (multicolinealidad)

Flujo (`feature_selection.py`): 33 features originales + 2 derivadas → filtro univariado C ≥ 0.55 →
**21 candidatas** → **VIF iterativo** (eliminar la peor hasta que todas tengan VIF ≤ 5) → **11 finales**.

Motivo: Cox es lineal y la multicolinealidad extrema (ver heatmap Spearman, fig 03) infla la varianza de
los coeficientes y rompe la convergencia. Features eliminadas por VIF > 5:

| Feature eliminada | VIF |
|-------------------|----:|
| `area_growth_rate_ha_per_h` | 32 024 |
| `radial_growth_m` | 2 558 |
| `centroid_displacement_m` | 201 |
| `radial_growth_rate_m_per_h` | 139 |
| `area_growth_rel_0_5h` | 46 |
| `centroid_speed_m_per_h` | 22 |
| `spread_bearing_deg` | 17 |
| `area_growth_abs_0_5h` | 12 |
| `low_temporal_resolution_0_5h` | 10 |
| `num_perimeters_0_5h` | 5.2 |

**Evidencia de que era necesario:** Cox con las 33 features da **C = 0.8655 ± 0.0986** (varianza enorme);
Cox con las 11 seleccionadas da **C = 0.9219 ± 0.0204** (varianza ~5× menor). La selección no solo
simplifica: estabiliza el modelo.

#### Por qué basta con dejar **una sola variable representante** (el mecanismo)

El Cox estima sus coeficientes β maximizando la **verosimilitud parcial**. Cuando dos o más features son
casi colineales, esa verosimilitud deja de tener un máximo nítido: se vuelve una **cresta plana** en la
dirección donde la suma de sus coeficientes es ~constante — los datos no pueden decidir **cómo repartir**
el efecto entre variables que se mueven juntas. En términos matriciales, la **matriz de información
(Hessiana)** se vuelve casi singular; su inversa —que es la covarianza de los coeficientes— explota, y con
ella los **errores estándar**. El resultado son coeficientes inestables (magnitudes enormes, a veces con
signos opuestos que se cancelan) y, en el límite de colinealidad perfecta (ρ = 1.0, varios pares del
heatmap fig 03), un modelo **no identificable** que ni siquiera converge.

**Dejar una sola representante elimina la redundancia, no la información.** Las variables colineales
codifican esencialmente la misma señal (mismo orden, casi el mismo span lineal): no aportan grados de
libertad nuevos, solo reparten una señal única entre varias columnas. Al conservar una sola, esa señal se
mantiene íntegra pero la verosimilitud recupera un máximo bien definido: la Hessiana vuelve a ser
invertible, los SE se contraen y el coeficiente queda estable e **interpretable** (un Hazard Ratio limpio
con su CI). Se pierde poquísimo poder predictivo porque lo descartado eran casi-duplicados.

**Evidencia directa** (experimento de escenarios de logs, `feature_selection.ipynb` §2.5, fig 22): sobre
la **data original cruda** probamos exactamente esto — quedarnos con **una** log como representante y
eliminar sus 14 correlacionadas (|Spearman| > 0.8):

| Escenario Cox (data original) | n_feat | C-index | C_std | IBS |
|-------------------------------|:------:|:-------:|:-----:|:---:|
| Original (todas, colineales) | 34 | 0.816 | 0.086 | 0.145 |
| **Con logs (1 representante, −14 colineales)** | 20 | **0.839** | **0.050** | 0.151 |
| Sin logs (quita logs, mantiene colineales) | 31 | 0.810 | 0.092 | 0.144 |

Quedarse con la representante **sube el C-index y reduce casi a la mitad la varianza entre folds**. En el
mismo experimento el **RSF no se inmuta** (C ≈ 0.948 en los tres escenarios) porque los
árboles parten variable a variable y son inmunes a la colinealidad: ese contraste es la prueba de que el
problema —y el beneficio de la poda— es **específico del Cox lineal**, no del dato.

> En una frase: la colinealidad no añade información, solo reparte la misma señal entre varias columnas y
> vuelve singular la estimación del Cox. Dejar una sola representante devuelve identificabilidad,
> estabilidad y un HR interpretable, sin sacrificar predicción. (Estas cifras usan las features originales
> crudas con L2 leve para aislar el efecto; el modelo Cox final usa las 11 features VIF ≤ 5 — §5.1 — y
> lifelines, con IBS aún mejor, 0.065.)

### 5.2 RSF — Permutation importance, y por qué el RSF es **invariante** a la selección

Criterio (`model_rsf.py`, `feature_selection.py`): eliminar features con importancia = 0 **y** std = 0
(ninguna señal en ninguna de las 15 permutaciones). 33 → **24 seleccionadas** (9 eliminadas):

```
dist_std_ci_0_5h, dist_change_ci_0_5h, dist_slope_ci_0_5h, closing_speed_m_per_h,
projected_advance_m, dist_accel_m_per_h2, along_track_speed,
estimated_hours_to_reach, relative_closing_rate
```

Motivo: `closing_speed_m_per_h` y sus derivados son **cero para todos los incendios lejanos** (event=0),
así que no aportan poder discriminativo. Como RSF (árboles) es inmune a la multicolinealidad, aquí el
criterio no es VIF sino "¿mueve la aguja del C-index?".

**Lo esencial: en el RSF la selección de features NO cambia el desempeño — el C-index se mantiene
invariante.** Mientras que en el Cox quitar las colineales fue *necesario* y bajó la varianza 30×
(§5.1), en el RSF quitar 9 features deja todo igual:

| Configuración RSF | n_feat | C-index CV | Std |
|-------------------|:------:|:----------:|:---:|
| Todas | 33 | 0.9485 | ±0.0061 |
| Seleccionadas | 24 | 0.9481 | ±0.0049 |
| **Δ** | **−9 feat** | **−0.0004 (ruido)** | varianza ≈ igual |

**Por qué el RSF es invariante (la razón):**

1. **Las features irrelevantes simplemente no se eligen.** El árbol parte una variable a la vez,
   escogiendo en cada nodo la mejor de un subconjunto aleatorio. Una feature sin señal **nunca gana un
   split** (o gana pocos, sin efecto sistemático): no degrada el modelo, queda ignorada. No hay
   coeficiente que estimar ni que se contamine.

2. **Las features redundantes/colineales son intercambiables, no dañinas.** Si dos variables codifican
   la misma señal, el árbol elige cualquiera de ellas y el corte resultante es **el mismo**. No existe el
   "reparto inestable del efecto" del Cox, porque en un árbol **no hay coeficientes** — hay **umbrales de
   partición**. La colinealidad no tiene dónde hacer daño.

3. **Consecuencia:** en el RSF la selección de features es por **parsimonia, velocidad e
   interpretabilidad** (menos columnas que explicar y permutar), **no por desempeño**.

> **Para la defensa — el contraste es la clave:** la poda del Cox (33 → 11) es **obligatoria** (sin ella
> el modelo es inestable y casi no converge); la del RSF (33 → 24) es **opcional y cosmética** (mismo
> C-index, misma varianza, modelo más limpio). El mismo dato produce dos regímenes opuestos según la
> naturaleza del modelo: **lineal (sensible a colinealidad) vs basada en árboles (invariante a ella)**.
> Que el RSF no se inmute es, además, una verificación cruzada de que las features eliminadas realmente
> no tenían señal.

### 5.3 Features derivadas que se intentaron y se descartaron

Se probaron 10 features nuevas en 2 rondas (ratios/escala y cinemáticas/físicas). **Ninguna** mejoró:

| Feature | Fórmula | C univariado | Razón de descarte |
|---------|---------|:------------:|-------------------|
| `projected_dist_12h` | `dist − speed·12 − ½·accel·144` | 0.91 | **VIF = 222 395** — combinación lineal exacta de dist+speed+accel |
| `etc_hours` | `dist / (closing_speed + ε)` | 0.48 | C < 0.55 |
| `eta_hours` | `dist / speed` | 0.48 | Centinela 999 distorsiona escala; C < 0.55 |
| `danger_index` | `area_growth × closing_speed` | 0.51 | C < 0.55 |
| `hour_sin / hour_cos` | `sin/cos(2π·hora/24)` | 0.54 | C < 0.55 |
| `accel_speed_ratio` | `accel / (speed + ε)` | 0.58 | No mejora CV (IBS sube) |
| `growth_shape` | `radial_growth / (area_growth + ε)` | 0.61 | No mejora CV |
| `expansion_vs_movement` | `radial_growth / (centroid_speed + ε)` | 0.60 | p = 0.95 multivariado; ΔC = +0.0003 |
| `acceleration_alarm` | `accel × area_growth` | — | Error de convergencia Cox (demasiados ceros) |

### 5.4 Cómo se validó el descarte (doble verificación)

El descarte inicial fue con Cox (lineal, sensible a VIF). Como los árboles pueden explotar ratios y
no-linealidades que Cox no ve, **se re-evaluaron las 10 features nuevas con RSF**:

| Modelo RSF | C-index CV | IBS CV |
|------------|:----------:|:------:|
| Base (24 features) | 0.9481 ± 0.0049 | 0.0304 |
| Aumentado (+10 = 34) | 0.9517 ± 0.0080 | 0.0307 |
| **Δ** | **+0.0036 (dentro del ruido)** | +0.0004 (peor) |

La permutation importance lo confirma: `projected_dist_12h` (C=0.91 univariado en Cox) obtiene solo
**+0.00054 ± 0.00186** en RSF (indistinguible de 0), porque `dist_min` / `log_dist_min` ya capturan
toda la señal de distancia. **Conclusión validada en ambos modelos: se conservan las features base.**

> Nota: estas comparaciones base-vs-aumentado se leen junto con el nested CV (§3.2). El ΔC de +0.0036 no
> solo es pequeño, está dentro del optimismo esperable por selección — otra razón para no añadir las
> features nuevas.

### Preguntas probables

- *¿No están botando información útil?* → No: cada feature descartada se validó. Las de VIF alto son
  redundantes (combinaciones lineales de otras); las de importancia 0 en RSF no mueven el C-index ni
  bajan el IBS. El ΔC de añadirlas todas es +0.0036, menor que la propia desviación estándar.
- *¿Por qué dos criterios distintos (VIF vs permutación)?* → Porque atacan problemas distintos: VIF mide
  redundancia lineal (relevante para Cox), permutación mide aporte predictivo real (relevante para RSF,
  que ignora la multicolinealidad).
- *¿Por qué descartar las colineales y dejar solo una mejora el Cox?* → Porque la colinealidad vuelve casi
  singular la Hessiana del Cox: los coeficientes y sus SE se disparan y deja de converger. Las colineales
  no aportan señal nueva, solo la reparten entre columnas; dejar una representante devuelve un máximo de
  verosimilitud único y estable (§5.1.1). El experimento de logs lo confirma: Cox sube de 0.810 a 0.839 y
  su varianza cae casi a la mitad, mientras el RSF —inmune a la colinealidad— no cambia.
- *¿Por qué `closing_speed` no sirve si "intuitivamente" debería?* → Es 0 para todos los incendios
  lejanos, así que no discrimina; la señal ya está contenida en la distancia.

---

## 6. Qué modelos y escenarios se probaron

**Decisión:** comparar tres familias de modelos de supervivencia y seleccionar por **C-index** (orden de
riesgo) e **IBS / Brier Score** (calibración), todo con **CV 5-fold estratificado por evento** (§3).
Los hiperparámetros, el espacio de búsqueda y **todos los scores por escenario** están consolidados en el
**Resumen al inicio**. Aquí va la descripción de cada modelo y las predicciones en test.

### 6.1 Los tres modelos

**a) Kaplan-Meier estratificado** — referencia no-paramétrica. Estratificado por cuartiles de
`log(dist_min_ci_0_5h)`; cada incendio de test hereda la curva de su cuartil. Limitación: solo asigna
4 valores posibles por horizonte, no discrimina dentro del cuartil → es referencia, no modelo final.

| Cuartil | n | Eventos | Dist. mediana | prob_12h | prob_24h | prob_48h | prob_72h |
|---------|---|---------|---------------|:--------:|:--------:|:--------:|:--------:|
| Q1 (cercano) | 56 | 56 (100%) | 2.1 km | 67.9% | 89.3% | 94.6% | 100% |
| Q2 | 55 | 13 (24%) | 7.5 km | 20.1% | 24.4% | 24.4% | 24.4% |
| Q3 | 55 | 0 | 100 km | 0% | 0% | 0% | 0% |
| Q4 (lejano) | 55 | 0 | 320 km | 0% | 0% | 0% | 0% |

**b) Random Survival Forest** (`scikit-survival`) — ✅ submission generado. Mejor C-index
(**0.9481 ± 0.0049** con 24 features; scores en el Resumen). Top features (permutation importance):
`dist_min_ci_0_5h` (0.0721) y `log_dist_min` (0.0648) concentran ~72 % de la señal.

**c) Cox PH** (`lifelines`, `penalizer=0.05`) — ✅ modelo activo, interpretable. C-index 0.9219 ± 0.0228,
IBS 0.0654 (~4× mejor que el modelo nulo de 0.25). Solo 2 features significativas: `log_dist_min`
(HR 0.217, protector) y `alignment_abs` (HR 1.412, riesgo).

### 6.2 Predicciones medias en test

| Horizonte | KM | Cox | RSF |
|-----------|:--:|:---:|:---:|
| 12h | 22.4% | 21.6% | 18.5% |
| 24h | 29.0% | 29.4% | 26.4% |
| 48h | 30.3% | 31.6% | 28.0% |
| 72h | 31.7% | **42.0%** | 29.8% |

> Cox diverge al alza a 72 h porque su función de supervivencia se extiende más en el tiempo, al no estar
> limitada por la distribución empírica de los árboles.

### 6.3 Verificación de monotonicidad

Se verificó que las probabilidades acumuladas (12→24→48→72 h) **no decrecen** por incendio, como exige
la coherencia de una función de supervivencia (decisión §1). Esto descarta predicciones incoherentes
antes de armar la submission.

### Preguntas probables

- *¿Por qué RSF si Cox es interpretable?* → Los usamos juntos: RSF da el mejor C-index (0.948) y es el
  submission; Cox da interpretabilidad (Hazard Ratios, significancia) y mejor calibración reportada
  (IBS). No compiten, se complementan.
- *¿Por qué no XGBoost/redes?* → Con n=221 y separación casi perfecta por distancia, modelos más
  complejos sobreajustarían sin ganancia; RSF ya satura el C-index en ~0.95.
- *¿Cómo sé que 0.948 no es optimista?* → Por el CV estratificado 5-fold + nested CV (§3.2) + adversarial
  validation (poco drift). El IBS 4× mejor que el modelo nulo respalda la calibración, no solo el orden.

---

## 7. Zona gris (5–15 km) — decisión pendiente ⚠️ NO IMPLEMENTADA

> **Estado: identificada y documentada; aún sin solución implementada.** Se deja por escrito como riesgo
> abierto y para fijar el plan de acción.

### El problema

Por la separación perfecta a 5 km, el train no contiene **ningún** incendio entre 5 y 15 km. El test sí:
**7 incendios (7.4 %) caen en esa zona gris**, todos además con `closing_speed = 0` y `alignment_abs = 0`.
Las predicciones del modelo ahí son **pura extrapolación** fuera del rango de entrenamiento, sin datos
que las respalden. Es el punto débil conocido del proyecto.

### Opciones consideradas (sin decidir aún)

| Opción | Idea | Riesgo / coste |
|--------|------|----------------|
| **A. No hacer nada** (baseline) | Dejar que el modelo extrapole con `log_dist_min`. | Honesto, pero la predicción en la zona gris no está validada. |
| **B. Datos sintéticos 5–15 km** | Generar incendios sintéticos en esa franja para que el modelo aprenda la transición. | Requiere *inventar* el mecanismo de etiquetado (event/tiempo) → inyecta un prior; debatible si lo permiten las reglas. |
| **C. Análisis de incertidumbre** | No reentrenar; cuantificar la incertidumbre de las predicciones en la zona gris (intervalos, sensibilidad). | Más defendible; no mejora la métrica pero acota el riesgo. |
| **D. Anclar a literatura externa** | Usar tasas reales de propagación de incendios para fijar el comportamiento esperado. | Aporta justificación física; depende de fuentes externas referenciadas. |

### Decisión actual y siguiente paso

- **Decisión:** por ahora se mantiene la **opción A** (extrapolación del modelo) como baseline, con la
  zona gris **explícitamente señalada** en EDA (figura 19) y reporte.
- **Pendiente:** evaluar la **opción C (incertidumbre)** como complemento de bajo riesgo y, solo si las
  reglas lo permiten y aporta señal validada, explorar la **opción B (sintéticos)**. La discusión de
  diseño de datos sintéticos quedó iniciada pero **no se ha implementado**.
- **Para la defensa:** lo importante es mostrar que el riesgo está *identificado y acotado*, no oculto.

### Preguntas probables

- *¿Qué hacen con los incendios de 5–15 km?* → Hoy el modelo extrapola con la distancia; lo tenemos
  identificado como zona sin datos de entrenamiento y documentado como riesgo. Estamos evaluando añadir
  un análisis de incertidumbre.
- *¿Por qué no generaron datos sintéticos?* → Es una opción sobre la mesa, pero generar etiquetas ahí
  implica asumir un mecanismo que el train no observa (un prior); antes de inyectarlo queremos confirmar
  que las reglas lo permiten y que aporta señal real, no ruido.

---

*Data Wizards · WiDS Datathon 2026 · 2026-06-12*
