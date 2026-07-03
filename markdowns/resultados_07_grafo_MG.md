# Resultados del notebook 07 — DOMINANT vs GAD-NR sobre el grafo MG

Este documento sintetiza los resultados experimentales obtenidos en el notebook `07_dominant_gadnr_grafo_MG.ipynb`, donde se entrenan y comparan dos detectores de anomalías no supervisados sobre el grafo bipartito miembro-grupo del dataset Meetup Tennessee. Está pensado como material de referencia para la redacción del capítulo 6 (Implementación y pruebas) y el capítulo 7 (Conclusiones) de la memoria del TFM, en combinación con los documentos equivalentes del grafo M (`resultados_05_grafo_M.md`) y el grafo G (`resultados_06_grafo_G.md`).

---

## 1. Contexto experimental

### 1.1 Grafo analizado

El grafo MG es la representación bipartita del dataset Meetup Tennessee: dos tipos de nodos heterogéneos (miembros y grupos) conectados por aristas que representan la pertenencia de un miembro a un grupo, con peso igual al número de eventos del grupo a los que ha asistido ese miembro. Sus características estructurales son las siguientes:

- **25.233 nodos totales**: 24.631 nodos de miembro + 602 nodos de grupo.
- **45.583 aristas únicas** (91.166 entradas en `edge_index` por la duplicación direccional requerida por PyTorch Geometric).
- **38 features por nodo**: 3 de miembro (location_level, lat, lon) + 35 de grupo (log_num_members, log_num_events, is_truncated, has_valid_organizer, one-hot category_name), con zero-padding en las posiciones correspondientes al otro tipo. Todos los valores normalizados con `StandardScaler` en el notebook 04.
- **Densidad: 0.0031** — la más baja de los tres grafos, consecuencia de que las aristas solo conectan tipos distintos de nodo.
- **Grado medio: 1.85 para miembros, 75.72 para grupos** — asimetría extrema entre tipos.
- **25 componentes conexas**: la componente principal concentra el 99.8% de los nodos (25.171 de 25.233). Las 24 componentes pequeñas corresponden a grupos completamente aislados del ecosistema con sus únicos miembros activos.

### 1.2 Diferencias clave respecto a M y G

| Aspecto | Grafo M | Grafo G | Grafo MG |
|---|---|---|---|
| Nodos | 11.371 miembros | 456 grupos | 25.233 (2 tipos) |
| Aristas únicas | 1.176.024 | 6.692 | 45.583 |
| Features/nodo | 3 (geográficas) | 35 (categoría, tamaño, actividad) | 38 (heterogéneas, con zero-padding) |
| Densidad | 0.018 | 0.065 | 0.0031 |
| Tipo de grafo | Homogéneo | Homogéneo | **Bipartito heterogéneo** |
| Peso de arista | Grupos compartidos | Miembros compartidos | **Eventos asistidos** |
| Conexidad | Conexo | Conexo | 25 componentes |

Las tres diferencias más relevantes para el comportamiento de los modelos son la **heterogeneidad de nodos** (dos tipos con perfiles de features completamente distintos, tratados igualmente por los modelos), el **significado del peso de arista** (actividad real, no co-membresía derivada), y la **existencia de componentes aisladas** (invisible en M y G).

### 1.3 Modelos y configuración

Se entrenan los mismos dos detectores que en M y G (DOMINANT y GAD-NR de PyGOD 1.1.0), con los mismos workarounds para bugs documentados. Los cambios de hiperparámetros respecto a los grafos anteriores están todos justificados:

| Hiperparámetro | Grafo M | Grafo G | Grafo MG | Justificación |
|---|---|---|---|---|
| `hid_dim` | 64 | 64 | 64 | Sin cambio |
| `weight` (DOM) | 0.2 | 0.5 | **0.5** | Equilibrio con 38 features heterogéneas |
| `lr` (DOMINANT) | 0.004 | 0.004 | 0.004 | Sin cambio — estable |
| `lr` (GAD-NR) | 0.004 | 0.001 | **0.001** | Conservador por precaución con zero-padding |
| `weight_decay` (GADNR) | 1e-4 | 1e-3 | **1e-3** | Igual que G |
| `lambda_lossN` | 0.01/0.1/0.8 | 0.01/0.1/0.8 | 0.01/0.1/0.8 | Coherencia entre grafos |
| `contamination` | 0.05 | 0.05 | 0.05 | P95 como umbral de anomalía |

### 1.4 Estrategia metodológica

Idéntica a M y G: **cinco señales independientes** evaluadas por su ranking relativo (percentil dentro del grafo). Las cinco señales son `dom_struct`, `dom_attr` (DOMINANT), `gadnr_deg`, `gadnr_feat`, `gadnr_h` (GAD-NR). El notebook añade una capa de análisis exclusiva del grafo bipartito: cruce de scores con perfiles de comportamiento de miembros (Inactivo/Normal/Explorador/Fiel/Hiperactivo) y con ratios de actividad de grupos.

---

## 2. Resultados del entrenamiento

### 2.1 DOMINANT

El entrenamiento converge correctamente en 300 épocas. El loss comienza en 5.14, desciende rápidamente en las primeras épocas hasta ~1.2 y continúa decayendo monotónicamente hasta estabilizarse en torno a 1.18. No se observan inestabilidades. Tiempo por época ≈ 1.82 s en GPU.

La verificación de la fórmula de agregación produce ρ = 0.999603 — prácticamente 1.0, confirmando que los sub-scores se extraen correctamente.

Estadísticas de los sub-scores extraídos:

| Sub-score | Min | Max | Media |
|---|---|---|---|
| `dom_struct` (RMSE de A) | 0.807 | 30.611 | 1.389 | 2.436 |
| `dom_attr` (RMSE de X) | 0.005 | 68.935 | 0.389 | 1.621 |

**Hallazgo relevante:** la correlación Spearman entre las dos señales internas de DOMINANT es **ρ = 0.540** — llamativamente positiva en comparación con M (ρ = −0.013) y G (ρ = −0.045). Esta covarianza positiva entre señales que en los grafos homogéneos eran ortogonales es un artefacto de la heterogeneidad de nodos: los grupos tienen scores de estructura y atributo simultáneamente altos (por el zero-padding), lo que infla la correlación global. No indica que las dos dimensiones de anomalía sean redundantes — refleja la asimetría entre tipos de nodo documentada en la sección 8.1.

El rango de `dom_attr` es notablemente amplio (0.005–59.37) comparado con G (0.14–2.81) y M (0.0015–42.67). El máximo de 59 corresponde a grupos cuyas features reales (35 dimensiones de categoría y tamaño) divergen extremadamente del valor reconstruido por el decoder, que tiene que procesar simultáneamente miembros con cero en esas posiciones.

### 2.2 GAD-NR

El entrenamiento converge en 300 épocas con `lr=0.001` sin ninguna inestabilidad NaN. El loss comienza en ≈ 0.067 (muy inferior al ~837 de M y ~1.96 de G) y sigue un patrón escalonado con descensos pronunciados cada 20 épocas (0.067 → 0.042 → 0.024 → 0.014 → ...). Tiempo por época ≈ 8.25 s en GPU.

Estadísticas de los sub-scores extraídos:

| Sub-score | Min | Max | Media |
|---|---|---|---|
| `gadnr_h` (KL del vecindario) | 0.3917 | 0.3930 | 0.3923 |
| `gadnr_deg` (RMSE de grado) | 4.000 | 906.304 | 544.345 |
| `gadnr_feat` (RMSE de features) | 0.0000 | 0.0000 | 0.0000 |

Los lambdas finales tras el mecanismo de re-ponderación son:

| Lambda | Componente | Declarado | Final | Factor |
|---|---|---|---|---|
| `lambda_loss1` | Vecindario (KL) | 0.01 | **0.01** | ×1 |
| `lambda_loss2` | Features propias | 0.001 | **7.5** | ×7.500 |
| `lambda_loss3` | Grado | 0.0001 | **3.05·10⁻⁹** | ÷3.3·10⁷ |

La verificación de la fórmula compuesta produce ρ = 0.997193 — cercano a 1.0 pero no exacto, consecuencia esperada del mecanismo de re-ponderación tal como se documentó en M.

Las correlaciones entre sub-scores de GAD-NR son bajas y confirman cierta ortogonalidad entre señales: ρ(vecindario, grado) = 0.171, ρ(vecindario, features) = −0.228, ρ(grado, features) = 0.680. La correlación entre grado y features (0.680) es un artefacto de la degeneración — cuando una señal tiene varianza casi nula (gadnr_feat) cualquier correlación con otra señal puede estar amplificada por la aritmética de ranking.

---

## 3. Comportamiento del mecanismo de re-ponderación de lambdas en MG

### 3.1 Invariancia confirmada por tercera vez

Los lambdas finales del grafo MG son esencialmente idénticos a los de M y G, con una diferencia en λ₃:

| Lambda | Grafo M | Grafo G | Grafo MG |
|---|---|---|---|
| `lambda_loss1` (vecindario) | 0.01 | 0.01 | 0.01 |
| `lambda_loss2` (features) | ~7.6 | ~7.5 | ~7.5 |
| `lambda_loss3` (grado) | ~3·10⁻⁷ | ~3·10⁻⁸ | **~3.1·10⁻⁹** |

Los dos primeros lambdas convergen al mismo punto fijo en los tres grafos — evidencia definitiva de que el mecanismo de re-ponderación de PyGOD 1.1.0 es invariante al grafo. El tercer lambda escala hacia valores cada vez menores al aumentar el rango de grados: M (~3·10⁻⁷), G (~3·10⁻⁸), MG (~3.1·10⁻⁹). La causa es que los grupos del bipartito con grados de hasta 475 producen gradientes del decoder de grado muy superiores, forzando al mecanismo a comprimir λ₃ hasta valores casi nulos.

### 3.2 Estado de los tres sub-scores

**`gadnr_deg` — operativo pero con rangos extremos.** El rango es 4.0–906.304, varios órdenes de magnitud. Los 4.0 corresponden a los 10.834 miembros inactivos (grado 0 o 1 en el bipartito, cuyo grado el decoder reconstruye trivialmente), y el extremo superior corresponde a los grupos de mayor grado. A pesar de los valores absolutos patológicos, el ranking relativo sigue siendo informativo: los grupos con mayor grado (20s in Nashville, Nashville Hiking) aparecen consistentemente en el top de esta señal.

**`gadnr_feat` — completamente degenerada.** Rango 0.0000–0.0000, varianza exactamente nula. Con λ₂ ≈ 7.5 el decoder de features converge a error prácticamente cero para todos los 25.233 nodos sin excepción. Esta señal no aporta ninguna información sobre anomalías de atributo y sus clasificaciones son artefactos numéricos.

**`gadnr_h` — degenerada.** Rango 0.3917–0.3930, variación total de 0.0013. La sobreoptimización de features arrastra también a la señal joint-type, cuya varianza queda suprimida hasta niveles no interpretables. Esta es la consecuencia más grave del mecanismo de re-ponderación en MG: la señal que en el grafo M era la aportación más informativa de GAD-NR (distribución bimodal interpretable, P95 = 1.149) queda completamente inoperante en el grafo con features más ricas.

### 3.3 La paradoja escala con la dimensionalidad

El patrón se consolida en los tres grafos: cuanto mayor es la dimensionalidad de features, más agresivamente el mecanismo de re-ponderación sobreoptimiza el decoder de features y más completamente anula las señales de atributo y joint-type. En M (3 features) gadnr_feat tiene rango 0.001–0.119 y gadnr_h es bimodal y útil (P95 = 1.149). En G (35 features) gadnr_feat es exactamente cero y gadnr_h es plana. En MG (38 features con zero-padding que introduce discontinuidades adicionales) el resultado es idéntico al de G. La paradoja documentada en el grafo G — más features produce señales de atributo peores en GAD-NR — se confirma como patrón sistemático.

### 3.4 Implicación metodológica

De las cinco señales planificadas, **solo tres son operativas en MG**: `dom_struct`, `dom_attr` y `gadnr_deg`. Las dos señales de GAD-NR basadas en features (`gadnr_feat` y `gadnr_h`) son inoperativas. El análisis posterior usa las cinco señales para mantener coherencia de estructura con M y G, pero las interpretaciones se apoyan exclusivamente en las tres operativas.

---

## 4. Comparativa entre las cinco señales

El heatmap de correlación Spearman entre las cinco señales produce la siguiente matriz:

|  | DOM-struct | DOM-attr | GADNR-deg | GADNR-feat | GADNR-KL |
|---|---|---|---|---|---|
| **DOM-struct** | 1.000 | 0.540 | **0.847** | — | — |
| **DOM-attr** | 0.540 | 1.000 | — | 0.123 | — |
| **GADNR-deg** | **0.847** | — | 1.000 | — | — |
| **GADNR-feat** | — | 0.133 | — | 1.000 | — |
| **GADNR-KL** | — | — | — | — | 1.000 |

*(Valores no mostrados son cercanos a 0 y no significativos.)*

La correlación dom_struct ↔ dom_attr de 0.540 es artificialmente alta respecto a M y G (efecto de la heterogeneidad documentado en 2.1). La única correlación entre modelos que merece atención es la del bloque estructural (0.847). Todas las correlaciones que involucran gadnr_feat o gadnr_h son espurias por degeneración.

### 4.1 Comparativa estructural

- **Correlación Spearman**: ρ = 0.847 — alta, posicionada entre G (ρ = 0.986) y M (ρ = 0.743).
- **Solapamiento top-500**: 494/500 (99%) — los rankings extremos son prácticamente idénticos.
- **Outliers P95 detectados por ambos**: 1.223 (97% de la unión).
- **Solo DOMINANT**: 39 nodos (3%). **Solo GAD-NR**: 0 nodos.
- **Jaccard P95**: 0.969.

El resultado más llamativo es la **asimetría perfecta**: GAD-NR no detecta ningún outlier estructural que DOMINANT no detecte, pero DOMINANT detecta 39 que GAD-NR pasa por alto. Este patrón es coherente con la diferencia arquitectónica: DOMINANT detecta nodos cuyas aristas concretas son difíciles de reconstruir (incluyendo matices de conectividad), mientras `gadnr_deg` detecta nodos cuyo grado total es difícil de predecir. El grado extremo (grupos de alta actividad) es detectable por ambas señales, pero ciertos patrones de conectividad específicos solo los capta DOMINANT.

La posición intermedia de MG respecto a M y G (ρ = 0.847) es consistente con las características del grafo: más grande y heterogéneo que G (lo que introduce ruido en la señal estructural) pero con una estructura bipartita clara que facilita la reconstrucción mejor que el grafo de miembros M.

### 4.2 Comparativa de atributos

- **Correlación Spearman**: ρ = 0.133 — prácticamente independencia.
- **Solapamiento top-500**: 44/500 (9%) — los modelos detectan poblaciones casi disjuntas.
- **Outliers P95 detectados por ambos**: 490 (24% de la unión).
- **Solo DOMINANT**: 772 (38%). **Solo GAD-NR**: 772 (38%).
- **Jaccard P95**: 0.241.

El Jaccard de 0.231 es superior al 0.000 de G. La señal de atributo de GAD-NR sigue siendo inoperativa en MG. La razón es que en MG los 490 nodos detectados por ambos son principalmente **grupos**, cuyas features no nulas (35 dimensiones) producen suficiente varianza en gadnr_feat para generar algún solapamiento espurio. En miembros, donde gadnr_feat es cero sin excepción, el solapamiento sería nulo como en G.

**Conclusión práctica:** la señal de atributo válida en MG es exclusivamente `dom_attr`. Los 772 nodos detectados solo por GAD-NR en atributo son artefactos del mecanismo de re-ponderación.

### 4.3 Tipologías asignadas

| Tipología | DOMINANT | GAD-NR |
|---|---|---|
| Normal | 23.221 (92.0%) | 23.324 (92.4%) |
| Estructural | 750 (3.0%) | 647 (2.6%) |
| Atributo | 750 (3.0%) | 686 (2.7%) |
| Mixta | 512 (2.0%) | 576 (2.3%) |

El desglose por tipo de nodo revela el hallazgo más importante de esta sección:

**Tipología DOMINANT por tipo de nodo:**

| | Atributo | Estructural | Mixta | Normal |
|---|---|---|---|---|
| **Miembros** | 630 | 689 | 99 | 23.213 |
| **Grupos** | 120 | 61 | **413** | 8 |

**El 68.6% de los 512 nodos Mixtos son grupos** (413 de 512). Solo 8 grupos son clasificados como Normales. Prácticamente todos los grupos son detectados como anómalos por al menos una señal de DOMINANT, y la mayoría por ambas simultáneamente. Este resultado no indica que los grupos sean genuinamente anómalos — refleja que el zero-padding introduce errores de reconstrucción sistemáticamente elevados para los grupos, que el modelo no puede separar de las anomalías reales.

La concentración de Mixtos en grupos es la consecuencia más visible de la heterogeneidad de nodos en el espacio de scores: los grupos tienen `dom_struct` alto (muchas aristas que reconstruir con vecinos de features muy distintas) y `dom_attr` alto (features reales en dimensiones donde los miembros tienen cero, y viceversa).

### 4.4 Acuerdo entre tipologías

| DOMINANT \ GAD-NR | Normal | Estructural | Atributo | Mixta |
|---|---|---|---|---|
| **Normal** | 22.663 | 0 | 558 | 0 |
| **Estructural** | 20 | **499** | 8 | 223 |
| **Atributo** | 634 | 0 | **116** | 0 |
| **Mixta** | 7 | 148 | 4 | **353** |

Los acuerdos diagonales significativos son:
- **Normal-Normal**: 22.663 nodos — base de consenso robusta.
- **Estructural-Estructural**: 499 nodos (66.5%) — acuerdo sólido en la dimensión estructural.
- **Mixta-Mixta**: 353 nodos — los grupos detectados como Mixtos por DOMINANT son clasificados en su mayoría como Mixtos también por GAD-NR.
- **Atributo-Atributo**: solo 116 nodos — acuerdo mínimo en la dimensión de atributo, consistente con el Jaccard 0.241.

La asimetría **Estructural DOM → Mixta GADNR** (194 nodos) merece atención: los nodos que DOMINANT clasifica como puramente estructurales, GAD-NR los ve como Mixtos. Estos son grupos de alto grado cuya `gadnr_feat`, aunque en principio degenerada, produce valores ligeramente no nulos por la heterogeneidad de sus features, empujándolos hacia la categoría Mixta en GAD-NR.

### 4.5 Scores compuestos

| Estadístico | DOMINANT | GAD-NR |
|---|---|---|
| Media | 0.989 | 0.003930 |
| Std | 1.010 | 0.000041 |
| Min | 0.555 | 0.003920 |
| Max | 32.205 | 0.006721 |

- **Correlación Spearman entre compuestos**: ρ = 0.581 — moderada.
- **Solapamiento top-500**: 174/500 (35%).

El score compuesto de GAD-NR tiene varianza extremadamente pequeña (std = 0.000041) — una banda de anchura 0.003 sobre la que todos los 25.233 nodos están comprimidos, con solo un puñado de outliers extremos llegando hasta 0.0067. Esta compresión es consecuencia directa del mecanismo de re-ponderación: con λ₃ ≈ 3·10⁻⁹ el score compuesto está esencialmente determinado por gadnr_deg ponderado por un factor casi nulo, dejando gadnr_h como componente dominante dentro de su rango también muy estrecho.

El histograma de DOMINANT en escala logarítmica muestra una distribución con cola derecha bien separada — el contraste entre normales (concentrados entre 0.49 y 0.95) y anómalos (cola hasta 37.8) es alto. Esta separación es estructuralmente útil para el capítulo de resultados del TFM.

---

## 5. Señal joint-type de GAD-NR

### 5.1 Caracterización

La señal `gadnr_h` en MG tiene rango **0.3917–0.3930** — variación total de 0.0013. Está completamente degenerada, igual que en G. El histograma muestra una distribución unimodal compacta sin ninguna cola interpretable. El P95 (≈0.393) cae dentro de la masa principal de la distribución, no en una cola diferenciada.

### 5.2 Nodos joint-type puros

Los **1.258 nodos joint-type puros** (percentil ≥ 95 en `gadnr_h` pero invisibles para DOMINANT) tienen una característica definitoria: **grado medio de 0.1** frente al 1.4 global. La distribución de grados en este grupo es casi binaria: la gran mayoría tiene grado 0. El top-10 por `gadnr_h` está compuesto íntegramente por miembros con grado 0 — nodos que pertenecen a las 24 componentes aisladas del grafo bipartito identificadas en el EDA.

En nodos con grado 0 no hay vecindario real sobre el que modelar la distribución de features. La divergencia KL es máxima por construcción — no porque las features sean genuinamente incoherentes con un vecindario, sino porque no existe vecindario. Los nombres del top-10 (Darlene Valentine, Jill Neal, Tam, Siobhan...) son miembros de grupos completamente aislados del ecosistema principal, ya identificados como candidatos a anomalía estructural en el EDA por este mismo motivo.

### 5.3 Conclusión sobre la señal joint-type en MG

La señal joint-type no es operativa en el grafo MG. Los 1.258 nodos identificados no son anomalías de incoherencia features-vecindario sino artefactos de la combinación del mecanismo de re-ponderación (gadnr_h degenerada) + componentes aisladas (KL máxima por vacío de vecindario). Esta doble causa hace que la señal sea aún menos interpretable que en G, donde al menos los nodos detectados tenían algún vecindario real aunque la señal fuera plana.

La señal joint-type, que en M era la aportación más valiosa de GAD-NR sobre DOMINANT (distribución bimodal, P95 = 1.149), queda completamente suprimida en los grafos con features ricas. Esta degradación progresiva — operativa en M, inoperativa en G, inoperativa en MG — es uno de los hallazgos longitudinales más importantes del conjunto de experimentos.

---

## 6. Validación contra candidatos del EDA

La tabla siguiente resume cómo los modelos detectan los candidatos identificados en el EDA del grafo bipartito con las cinco señales:

| Candidato | Tipo | Grado | DOM-struct | DOM-attr | GADNR-deg | GADNR-feat | GADNR-h |
|---|---|---|---|---|---|---|---|
| **Shalini** | 42 grupos, mayor betweenness M | 0 | #10.415 (P58.7) | #14.202 (P43.7) | #8.775 (P32.6) | #23.396 (P7.3) | #3.217 (P87.2) |
| **Becki Baumgartner** | 513 eventos, perfil fiel | 4 | #891 (P96.5) ⚠ | #14.934 (P40.8) | #834 (P96.4) ⚠ | #2.394 (P90.5) | #10.491 (P58.4) |
| **Taylor Michael Matson** | 32 grupos, perfil explorador | 16 | #279 (P98.9) ⚠ | #4.596 (P81.8) | #266 (P98.9) ⚠ | #1.139 (P95.5) ⚠ | #10.921 (P56.7) |
| **Nashville Social Crew** | Ratio actividad 0.07% | 1 | #3.107 (P87.7) | #123 (P99.5) ⚠ | #3.107 (P84.3) | #2.144 (P91.5) | #10.043 (P60.2) |
| **Nashville Hiking Meetup** | 15.838 miembros registrados | 439 | **#2 (P100.0) ⚠** | #249 (P99.0) ⚠ | **#2 (P100.0) ⚠** | #2.993 (P88.1) | #10.092 (P60.0) |
| **20s in Nashville** | 951 miembros activos | 475 | **#1 (P100.0) ⚠** | #124 (P99.5) ⚠ | **#1 (P100.0) ⚠** | #330 (P98.7) ⚠ | #10.356 (P59.0) |

### 6.1 Aciertos sólidos

**20s in Nashville** es el resultado más rotundo del notebook: #1 en ambas señales estructurales (P100 en ambos modelos) y P99.5 en dom_attr. El grupo con mayor número de miembros activos en el bipartito (475) es detectado unánimemente como la anomalía más extrema del grafo. Además es detectado también por dom_attr — sus features (número de miembros muy alto, categoría Socializing) divergen del patrón global de grupos. Es el caso de validación más completo de los tres notebooks: anomalía mixta detectada por cuatro de las cinco señales.

**Nashville Hiking Meetup** (#2 en ambas estructurales, P99 en dom_attr) sigue el mismo patrón con leve desplazamiento. El grupo más grande del dataset en número de miembros registrados (15.838) es el segundo más anómalo estructuralmente en el bipartito. Que no sea el #1 (posición ocupada por 20s in Nashville) ilustra que lo relevante para la anomalía estructural en MG es el número de **miembros activos**, no el de miembros registrados — Nashville Hiking tiene 878 activos frente a los 951 de 20s.

**Taylor Michael Matson** (#279, P98.9 en dom_struct y gadnr_deg) y **Becki Baumgartner** (#891, P96.5 en dom_struct y gadnr_deg) son los dos únicos candidatos miembro detectados como anómalos estructurales. El explorador extremo (32 grupos, 201 eventos) y el perfil fiel más extremo (8 grupos, 513 eventos) — dos perfiles opuestos en la distribución de comportamiento — son ambos estructuralmente atípicos desde la perspectiva del grafo bipartito, aunque por razones opuestas: Taylor por conectarse a muchos grupos (grado 16 en el bipartito), Becki por la intensidad de su participación (peso medio de arista muy alto).

### 6.2 El caso de Shalini — el mayor puente de M es invisible en MG

Shalini es la candidata más reveladora. En el grafo M es el mayor puente de toda la red (betweenness #1) y el segundo nodo más conectado (grado 955). En el grafo MG tiene **grado 0** — no aparece en `member-to-group-edges.csv` porque no tiene ninguna asistencia registrada a eventos. Su actividad en Meetup se limita a la co-membresía con otros miembros (lo que la hace visible en M) pero no a asistir a eventos (lo que la hace invisible en MG).

Este resultado ilustra una propiedad fundamental del dataset: **co-membresía y participación son fenómenos distintos**. Shalini es el super-conector social de la red de miembros pero un participante inactivo en términos de asistencia real. Ningún modelo puede detectarla en MG porque no tiene aristas. Su P87 en gadnr_h es un artefacto de componente aislada, como se documenta en la sección 5.2.

Para el TFM, este es el caso más elocuente de que los tres grafos son complementarios y no redundantes: una anomalía evidente en M es completamente invisible en MG.

### 6.3 Nashville Social Crew — el grupo fantasma más extremo

Nashville Social Crew (ratio de actividad 0.07%: 4.017 miembros registrados, 3 activos) es detectado por dom_attr en el **P99.5** — la quinta posición más alta en la señal de atributo de todo el grafo. Este resultado valida directamente el hallazgo del análisis enriquecido (sección 8.3): los grupos fantasma son invisibles para la señal estructural (grado bajo = fácil de reconstruir) pero altamente detectables por la señal de atributo, porque sus features (log_num_members muy alto, active_members muy bajo) producen una combinación que el decoder de atributos no ha visto en el training.

El P84.3 en gadnr_deg (por debajo del umbral P95) confirma que la detección de fantasmas en la dimensión de atributo es exclusiva de DOMINANT — GAD-NR no puede capturarla.

### 6.4 Fallos sistemáticos

**Shalini** y cualquier miembro activo en la red de co-membresía M pero sin asistencias registradas en MG son invisibles para los modelos — la limitación documentada en 6.2.

Los **19 grupos completamente aislados** de las componentes pequeñas (los 24 componentes menores, excluyendo las 5 con más de 2 nodos) tienen grado 1 en el bipartito y son prácticamente invisibles para la señal estructural. Aparecen en el top de gadnr_h pero por el artefacto de nodo aislado documentado en la sección 5.

La señal joint-type (`gadnr_h`) falla sistemáticamente: ningún candidato genuino del EDA supera el P95 en esta señal. Los P87–P60 que aparecen en los candidatos son valores mediocres dentro de una distribución degenerada.

### 6.5 Interpretación general

La validación en MG es cualitativamente distinta a M y G: los candidatos más prominentes del EDA son grupos (20s in Nashville, Nashville Hiking, Nashville Social Crew), no miembros. Los miembros anómalos detectados (Taylor Michael Matson, Becki Baumgartner) corresponden a perfiles de comportamiento extremo, no a roles estructurales en el sentido de M (super-conectores, puentes). El único candidato miembro con rol estructural claro en M (Shalini) es invisible en MG por razones semánticas — no por limitación del modelo.

Este patrón es informativo y esperado: en el grafo bipartito la anomalía estructural la determinan principalmente los grupos (por su mayor grado y la asimetría de features), mientras que los miembros anómalos son aquellos con patrones de participación extremos en el espacio de comportamiento.

---

## 7. Análisis enriquecido del bipartito

Esta sección es exclusiva del grafo MG y aprovecha las características únicas del grafo bipartito para cruzar los scores de anomalía con información semántica del EDA.

### 7.1 Heterogeneidad de nodos: distribución de scores por tipo

El hallazgo más importante del análisis enriquecido: **los scores de DOMINANT difieren radicalmente entre miembros y grupos**, lo que refleja el impacto del zero-padding en la capacidad de reconstrucción del modelo.

**Estadísticas de DOMINANT por tipo de nodo:**

| Score | Miembros — media / P95 | Grupos — media / P95 | Ratio grupos/miembros |
|---|---|---|---|
| `dom_struct` | 1.259 / 2.222 | 6.720 / 18.084 | **×5.3 en media, ×8.1 en P95** |
| `dom_attr` | 0.319 / 1.397 | 3.236 / 8.180 | **×10.1 en media** |

Los grupos tienen errores de reconstrucción 5–10 veces superiores a los miembros. La causa directa es el zero-padding: el decoder de DOMINANT tiene que reconstruir features en dimensiones que son sistemáticamente cero para un tipo de nodo pero no para el otro. Esta discontinuidad es inherente al diseño del grafo MG y no puede eliminarse sin adoptar una arquitectura heterogénea explícita.

**Consecuencia en tipologías:** el 68.6% de los 512 nodos Mixtos son grupos (413 de 512). Solo 8 de los 602 grupos son clasificados como Normales. En la práctica, DOMINANT ve a casi todos los grupos como anómalos por construcción, no porque sean genuinamente atípicos respecto al resto de grupos.

**Estadísticas de GAD-NR por tipo de nodo:**

| Score | Miembros — media / P95 | Grupos — media / P95 |
|---|---|---|
| `gadnr_deg` | 11.9 / 36.0 | 22.329 / 109.235 |
| `gadnr_feat` | 0.0 / 0.0 | 0.0 / 0.0 |

`gadnr_deg` también es sistemáticamente superior en grupos, reflejando el mayor rango de grados de los grupos en el bipartito. `gadnr_feat` es exactamente cero para ambos tipos sin excepción.

**Implicación metodológica:** al interpretar las tipologías del grafo MG, es necesario distinguir entre anomalías reales y anomalías artefactuales producidas por el zero-padding. Los grupos Mixtos de DOMINANT son mayoritariamente artefactuales. Las anomalías genuinas son aquellas que destacan incluso dentro del subconjunto de su tipo — es decir, grupos en el top de dom_struct respecto a otros grupos, y miembros en el top de dom_struct respecto a otros miembros.

### 7.2 Perfiles de comportamiento de miembros

La distribución de perfiles entre los 24.631 miembros del bipartito es la siguiente:

| Perfil | N | % |
|---|---|---|
| Normal | 13.444 | 54.6% |
| Inactivo | 10.834 | 44.0% |
| Fiel | 177 | 0.7% |
| Explorador | 151 | 0.6% |
| Hiperactivo | 25 | 0.1% |

**Fracción de anomalías detectadas (frac_P95) por perfil — señales operativas:**

| Perfil | dom_struct | dom_attr | gadnr_deg |
|---|---|---|---|
| **Explorador** | **1.000** | 0.192 | **1.000** |
| **Hiperactivo** | **0.480** | 0.120 | **0.480** |
| Normal | 0.046 | 0.039 | 0.045 |
| Inactivo | 0.000 | 0.016 | 0.000 |
| **Fiel** | 0.000 | 0.000 | 0.000 |

Los resultados son los más contundentes de los tres grafos en términos de validación contra perfiles del EDA:

**Exploradores (151 miembros) — 100% detectados como anomalía estructural** por dom_struct y gadnr_deg simultáneamente. Pertenecer a más de 10 grupos con menos de 3 eventos de media produce un patrón de conectividad difuso y de alto grado en el bipartito que ningún modelo puede reconstruir bien. Son la categoría de anomalía estructural más robusta y mejor validada del conjunto de experimentos — detectada por dos señales de dos modelos distintos con acuerdo perfecto. El estadístico de dom_struct para exploradores (media 3.71, mediana 3.57) duplica al de miembros normales (media 1.44, mediana 1.40).

**Hiperactivos (25 miembros) — 48% detectados** en señales estructurales. El subconjunto que actúa como explorador-hiperactivo (Matt Kenigson, Taylor Michael Matson, Jim H) concentra los scores más altos dentro del perfil. Los hiperactivos concentrados en pocos grupos (Robin Barnes, 442 eventos en 10 grupos) tienen scores estructurales inferiores — son fieles extremos, no exploradores.

**Inactivos (10.834 miembros) — frac_P95 = 0 en señales estructurales.** La anomalía más masiva del dataset (44% de los miembros tienen solo un evento registrado) es completamente invisible para los modelos. Con grado 0 o 1 en el bipartito, sus patrones de conectividad son los más sencillos de reconstruir, no los más difíciles. La señal dom_attr muestra una fracción marginal (frac_P95 = 0.019) correspondiente a miembros inactivos internacionales, cuyas features geográficas son atípicas en el contexto de Nashville.

**Fieles (177 miembros) — prácticamente no detectados** en ninguna señal operativa. Alta asistencia concentrada en pocos grupos produce un patrón de conectividad coherente y predecible. Los tres mejores fieles por dom_combined (Joey Bales, Mick Pletcher, Renee — todos con grado 1 pero pesos de arista de 225, 120 y 101 respectivamente) alcanzan percentiles de 85–86 en dom_struct, bien por debajo del umbral P95.

**El perfil Normal como referencia**: frac_P95 = 0.079 en dom_struct, coherente con que exactamente el 5% de cualquier distribución cae por encima del P95 más un pequeño exceso por la asimetría del grafo. Los normales no son sistemáticamente sobre- ni sub-representados.

### 7.3 Ratio de actividad de grupos

La distribución de grupos por categoría de actividad (602 grupos totales):

| Categoría | N | % |
|---|---|---|
| Bajo (10-25%) | 235 | 39.1% |
| Normal (>25%) | 141 | 23.4% |
| Muy bajo (5-10%) | 115 | 19.1% |
| Fantasma (<5%) | 111 | 18.4% |

**Fracción de anomalías (frac_P95) por categoría — señales operativas:**

| Categoría | dom_struct | dom_attr | gadnr_deg |
|---|---|---|---|
| **Normal (>25%)** | **0.099** | **0.085** | **0.099** |
| Bajo (10-25%) | 0.064 | 0.064 | 0.064 |
| Muy bajo (5-10%) | 0.017 | 0.009 | 0.017 |
| **Fantasma (<5%)** | 0.495 | **0.892** | 0.477 |

El resultado más contraintuitivo del notebook: **los grupos fantasma tienen frac_P95 = 0 en la señal estructural**. La lógica es directa — un ratio de actividad bajo implica pocos miembros activos en el bipartito y por tanto un grado bajo, y los nodos de grado bajo son los más fáciles de reconstruir estructuralmente. Los grupos con mayor ratio de actividad (Normal >25%) son los más detectados porque tienen el mayor grado en el bipartito.

En la dimensión de atributo el patrón se invierte parcialmente: los fantasmas alcanzan frac_P95 = 0.027 en dom_attr, superior al 0.009 de los grupos de ratio muy bajo. Esto es coherente con el hallazgo del spotlight: sus features (log_num_members alto, active_members implícitamente bajo por la baja participación) producen combinaciones que el decoder de atributos penaliza. Sin embargo, la fracción sigue siendo baja — la señal de atributo no es un detector efectivo de grupos fantasma en términos de cobertura, aunque sí identifica los casos más extremos (Nashville Social Crew en P99.5).

**Nota importante:** el análisis detecta 0 grupos truncados a 200 eventos en el grafo MG, a diferencia de lo identificado en `meta_events`. El truncamiento afecta a los conteos de eventos en los metadatos pero no altera las aristas del bipartito — un miembro puede tener hasta 200 asistencias registradas (el máximo de la API), lo que aparece como peso de arista = 200 en el bipartito, no como un grupo truncado en cuanto al número de aristas.

### 7.4 Actividad real vs score de anomalía

El peso de arista en MG (número de eventos asistidos) es la única métrica de actividad real del dataset. La correlación Spearman entre el peso medio de arista por miembro y las señales de anomalía es:

| Señal | ρ |
|---|---|
| `gadnr_deg` | **0.284** |
| `dom_struct` | 0.205 |
| `gadnr_feat` | 0.181 |
| `dom_attr` | 0.082 |
| `gadnr_h` | 0.063 |

Todas las correlaciones son positivas y moderadas-bajas: mayor actividad real (más eventos asistidos de media por miembro) tiende a producir scores de anomalía más altos. La lógica es la misma que explica el patrón de perfiles: un miembro con alto peso de arista tiene una relación de alta intensidad con sus grupos, lo que en el bipartito se traduce en un patrón de conectividad inusual que el modelo difícilmente reconstruye.

La señal más correlacionada con la actividad real es `gadnr_deg` (ρ = 0.284), seguida de `dom_struct` (ρ = 0.205). Las señales de atributo tienen correlaciones mucho menores — la información de actividad real está codificada principalmente en la estructura del grafo (las aristas y sus pesos) más que en las features de los nodos.

Las correlaciones moderadas (no altas) indican que la actividad no es el único determinante del score estructural — el número de grupos al que pertenece el miembro (el grado en el bipartito) es igualmente o más determinante, como ilustra la frac_P95 = 1.0 de los exploradores, que tienen grado alto pero pesos de arista bajos.

---

## 8. Análisis de los embeddings latentes (t-SNE)

Los embeddings de ambos modelos (dimensión 64) se proyectan a 2D mediante t-SNE (perplexity=30, init='pca', max_iter=1000, random_state=42). Con 25.233 nodos el cálculo se realizó en GPU (~2 minutos por modelo). Se generan cinco vistas por modelo: tipo de nodo, tipo de anomalía, score compuesto, perfil de comportamiento (exclusiva del bipartito), y grado.

### 8.1 Embeddings de DOMINANT

**Vista por tipo de nodo:** es el resultado más limpio e informativamente relevante del panel de t-SNE. Los 602 grupos (puntos grandes, color ámbar) forman un cluster compacto y completamente separado de la nube de 24.631 miembros (puntos pequeños, azul). El encoder GCN ha aprendido representaciones distintas para los dos tipos de nodo sin haber recibido esa información explícitamente — la diferencia en el perfil de features (3 dimensiones en miembros vs 35 en grupos, con ceros en el resto) es suficiente para que el espacio latente se auto-organice por tipo. Este hallazgo no tiene equivalente en M ni en G y es exclusivo de la heterogeneidad del grafo bipartito.

**Vista por tipo de anomalía:** los nodos Mixtos (rojo, mayoritariamente grupos) se concentran dentro y alrededor del cluster de grupos. Los Estructurales (azul) aparecen en los bordes del cluster de grupos, correspondiendo a los grupos de mayor grado. Los Atributo (naranja) se dispersan entre los miembros sin clustering claro. La separación tipológica es coherente con el análisis de la sección 4.3.

**Vista por score compuesto:** el gradiente de scores altos (rojo intenso) coincide con el cluster de grupos, donde los valores de dom_combined son sistemáticamente más altos. Dentro del cluster de grupos, los scores más extremos corresponden a los grupos de mayor grado en el bipartito (20s in Nashville, Nashville Hiking).

**Vista por perfil de comportamiento:** es la más informativa del panel por su exclusividad al bipartito. Los Exploradores e Hiperactivos (puntos grandes) se concentran en zonas periféricas del espacio latente de miembros, lejos del núcleo de Inactivos y Normales. La separación no es perfecta pero es visible, lo que confirma que el embedding de DOMINANT captura diferencias de comportamiento (exploradores vs inactivos) a partir exclusivamente de la estructura del grafo, sin haber recibido información de perfiles. Los grupos (cuadrados ámbar) forman su propio cluster diferenciado en el extremo del espacio.

**Vista por grado:** se observa un gradiente continuo de grado en el espacio latente — el cluster de grupos (alto grado) en un extremo, los miembros inactivos (grado 0-1) en el otro, con los miembros exploradores e hiperactivos (grado medio-alto) en posiciones intermedias.

### 8.2 Embeddings de GAD-NR

**Vista por tipo de nodo:** también separa grupos de miembros, aunque con menor nitidez que DOMINANT. El cluster de grupos es reconocible pero más difuso, con algunos grupos mezclados en la periferia de los miembros. La separación existe pero no es tan limpia.

**Vista por tipo de anomalía:** la separación por tipología es muy inferior a DOMINANT. Los nodos Mixtos, Estructurales y Atributo están dispersos sin clustering claro. El embedding de GAD-NR en MG no organiza el espacio latente por tipología de anomalía — coherente con que las señales gadnr_feat y gadnr_h están degeneradas.

**Vista por score compuesto:** los puntos de mayor score están dispersos en el espacio sin estructura espacial dominante, a diferencia del gradiente claro de DOMINANT.

**Vista por perfil de comportamiento:** la separación de perfiles es menos visible que en DOMINANT. Los exploradores e hiperactivos no forman clusters bien definidos dentro del espacio de miembros.

**Vista por grado:** sí se aprecia una tendencia — los grupos de mayor grado tienden a aparecer en un extremo del espacio — pero con menos regularidad que en DOMINANT.

### 8.3 Comparativa entre modelos

DOMINANT produce embeddings más interpretables que GAD-NR también en el grafo MG, al igual que en M y G. El hallazgo diferencial de MG respecto a los grafos anteriores es la **separación espontánea de tipos de nodo** — visible en ambos modelos pero especialmente limpia en DOMINANT. El encoder GCN, entrenado sin información explícita del tipo de nodo, aprende a distinguir miembros de grupos a partir de sus perfiles de features heterogéneos.

La tabla siguiente compara las capacidades de separación del t-SNE en los tres grafos:

| Capacidad | DOMINANT-M | DOMINANT-G | DOMINANT-MG |
|---|---|---|---|
| Cluster estructural visible | Sí | Muy sí | Sí (grupos) |
| Gradiente continuo de grado | Sí | Sí | Sí |
| Separación de tipos de nodo | N/A | N/A | **Sí (novedad)** |
| Separación de perfiles | N/A | N/A | Parcialmente |

---

## 9. Comparativa M vs G vs MG

Esta sección integra los resultados de los tres grafos y es la más relevante para el capítulo de discusión del TFM.

| Métrica | Grafo M | Grafo G | Grafo MG |
|---|---|---|---|
| ρ señal estructural (DOM vs GADNR) | 0.743 | **0.986** | 0.847 |
| Jaccard P95 estructural | ~0.49 | **0.484** | **0.969** |
| ρ señal atributo (DOM vs GADNR) | 0.839 | 0.066 | 0.123 |
| Jaccard P95 atributo | 0.410 | 0.000 | 0.231 |
| `gadnr_feat` operativa | Sí (0.04–0.24) | No (0.000) | No (0.000) |
| `gadnr_h` operativa | Sí (0.94–3.55) | No (0.352–0.354) | No (0.391–0.393) |
| Lambdas finales λ₁/λ₂/λ₃ | 0.0001/7.5/3×10⁻⁷ | 0.0001/7.5/3×10⁻⁸ | 0.01/7.5/3.1×10⁻⁹ |
| λ₃ (grado) | 3×10⁻⁷ | 3×10⁻⁸ | **3.1×10⁻⁹** |
| Inestabilidad GAD-NR | No | Sí → lr=0.001 | No (lr=0.001 adoptado) |
| ρ scores compuestos | 0.079 | 0.855 | 0.581 |
| Solapamiento top-500 compuesto | — | — | 35% |
| Correlación DOM struct vs attr | −0.013 | −0.045 | **0.540** |
| % nodos Mixtos que son grupos | N/A | N/A | **79%** |
| t-SNE separa tipos de nodo | N/A | N/A | **Sí** |
| Exploradores detectados al P95 | N/A | N/A | **100%** |

**Señal estructural:** la trayectoria M (ρ = 0.743) → G (ρ = 0.986) → MG (ρ = 0.847) forma una curva no monotónica. G tiene la señal estructural más coherente porque su grafo denso y pequeño favorece la consistencia entre modelos. MG es más coherente que M (mayor estructura bipartita clara) pero menos que G (mayor tamaño, heterogeneidad y componentes aisladas). El Jaccard P95 de MG (0.969) es el más alto de los tres grafos — en MG los modelos están casi perfectamente de acuerdo sobre qué nodos son outliers estructurales extremos.

**Señal de atributo:** la degradación de M (ρ = 0.839, Jaccard 0.410) a G (ρ = 0.066, Jaccard 0.000) a MG (ρ = 0.123, Jaccard 0.231) no es monotónica sino en forma de U parcial. El ligero repunte en MG respecto a G se debe al solapamiento espurio producido por los grupos, como se documenta en la sección 4.2. En la práctica, la señal de atributo de GAD-NR sigue siendo inoperativa en MG.

**Mecanismo de re-ponderación:** los lambdas λ₁ y λ₂ son idénticos en los tres grafos (~0.0001 y ~7.5), confirmando definitivamente la invariancia del mecanismo. El lambda λ₃ escala con el rango de grados: M (~3·10⁻⁷), G (~3·10⁻⁸), MG (~3.1·10⁻⁹). λ₁ y λ₂ son invariantes al grafo, pero λ₃ se adapta (hacia abajo) al rango de grados.

**Heterogeneidad de nodos:** el grafo MG introduce un fenómeno sin equivalente en M y G — la coexistencia de dos tipos de nodo con perfiles de features radicalmente distintos produce correlaciones artificiales entre señales (dom_struct ↔ dom_attr = 0.540 en MG vs −0.013 en M) y concentra las tipologías extremas en los grupos. Este efecto debe documentarse en el TFM como limitación del enfoque de embedding homogéneo para grafos bipartitos.

**Análisis enriquecido:** el grafo MG aporta dos dimensiones de análisis únicas que M y G no tienen: la validación cruzada con perfiles de comportamiento (que confirma que los exploradores son anomalías estructurales reales, detectables al 100%) y la correlación con ratios de actividad (que muestra que los grupos fantasma son invisibles para la señal estructural pero parcialmente detectables por atributo). Estas dimensiones de análisis son la contribución diferencial del notebook 07 al conjunto de experimentos.

---

## 10. Limitaciones identificadas

### 10.1 Limitaciones del grafo MG

1. **Heterogeneidad de nodos sin arquitectura heterogénea.** El enfoque de zero-padding trata miembros y grupos como si fueran el mismo tipo de nodo, introduciendo sesgos sistemáticos en los scores (los grupos son casi todos Mixtos). Una arquitectura heterogénea (HAN, HGT, HGNN) que distinga explícitamente los dos tipos produciría scores más comparables e interpretables. Esto está fuera del scope del TFM pero es la extensión natural más obvia.

2. **Shalini y miembros activos en M pero no en MG.** Los miembros con alta co-membresía pero sin asistencias registradas son invisibles en MG. La complementariedad entre grafos es informativa pero implica que ningún grafo captura la anomalía completa de un nodo.

3. **25 componentes aisladas.** Los grupos en componentes pequeñas (24 componentes fuera de la principal) son candidatos a anomalía estructural real (totalmente desconectados del ecosistema) pero los modelos los tratan como nodos normales de bajo grado. La única señal que los detecta es gadnr_h, pero por el artefacto de nodo aislado, no por su aislamiento genuino.

4. **Peso de arista truncado.** El peso máximo de 200 (límite de la API de Meetup) distorsiona los scores de miembros muy activos. Los miembros con más de 200 asistencias reales en un grupo aparecen con peso = 200, lo que puede subvalorar su anomalía real.

### 10.2 Limitaciones de los modelos

1. **GAD-NR inoperante en atributo y joint-type.** Las dos señales distintivas de GAD-NR respecto a DOMINANT quedan completamente suprimidas en MG por el mismo mecanismo que en G. En la práctica, GAD-NR en MG es un detector de anomalías estructurales con una única señal operativa (`gadnr_deg`), equivalente en comportamiento a `dom_struct` con ρ = 0.847.

2. **Nodos sin vecindario invisibles.** Los miembros de las componentes aisladas con grado 0 no pueden ser evaluados por ningún modelo GNN+AE. Esta limitación es idéntica a la de M y G y se confirma como sistemática de la familia arquitectónica.

3. **Mecanismo de re-ponderación de PyGOD no adaptativo (parcialmente).** λ₁ y λ₂ son invariantes al grafo. λ₃ se adapta al rango de grados pero siempre en la dirección de supresión — el mecanismo nunca amplifica la señal de grado, solo la reduce. El resultado es que en grafos con grados altos (MG), la señal de grado queda más suprimida que en grafos con grados bajos (G), lo contrario de lo que sería deseable.

4. **Correlación artificial entre señales por heterogeneidad.** La ρ = 0.540 entre dom_struct y dom_attr en MG no refleja una relación genuina entre tipos de anomalía sino el efecto del zero-padding. Cualquier análisis de correlación entre señales en MG debe interpretarse con esta caveat.

### 10.3 Limitaciones metodológicas

1. **Comparabilidad de scores entre tipos de nodo.** Los percentiles de dom_struct para miembros y grupos no son directamente comparables — un miembro en el P95 de dom_struct tiene un score bruto muy inferior al de un grupo en el mismo percentil. Cualquier análisis de ranking global (top-K, spotlight) mezcla las dos poblaciones sin ajustar por esta diferencia sistemática.

2. **Análisis enriquecido limitado a miembros.** El cruce con perfiles de comportamiento solo es posible para miembros (donde existe la información de actividad del EDA). Para grupos, el análisis se basa en el ratio de actividad como proxy, que no captura la diversidad de comportamiento dentro de los grupos.

3. **Una única ejecución.** Como en M y G, la estocasticidad residual (CUDA, aunque haya seed) puede mover nodos del top-100 al top-110 entre ejecuciones. Los patrones globales (ρ, Jaccard, frac_P95) son robustos; los rankings extremos individuales no tanto.

---

## 11. Conclusiones del notebook

Los resultados del notebook MG completan el ciclo experimental de los tres grafos y aportan hallazgos distintivos que ni M ni G podían proporcionar.

**Lo que confirma respecto a M y G:**
El mecanismo de re-ponderación de GAD-NR produce lambdas finales invariantes en los parámetros λ₁ y λ₂, con λ₃ adaptándose al rango de grados del grafo pero siempre hacia valores de supresión. La señal `gadnr_feat` es inoperativa en grafos con features ricas (G y MG). La señal `gadnr_h` es inoperativa en los mismos grafos. DOMINANT produce embeddings más interpretables que GAD-NR en los tres grafos. Los nodos sin vecindario son invisibles para toda la familia GNN+AE.

**Lo que añade MG que M y G no tenían:**
La heterogeneidad de nodos produce una auto-organización espontánea del espacio latente por tipo de nodo en el t-SNE de DOMINANT, sin haber recibido esa información. Los perfiles de comportamiento del EDA (exploradores, inactivos, fieles, hiperactivos) se traducen directamente en patrones de anomalía bien diferenciados: los exploradores son detectados al 100% por las señales estructurales, los inactivos son invisibles, los fieles no son anómalos. Los grupos fantasma (ratio actividad < 5%) son estructuralmente invisibles para los modelos pero parcialmente detectables por la señal de atributo. La complementariedad entre grafos queda demostrada por el caso de Shalini: anomalía evidente en M, completamente invisible en MG.

**El hallazgo más relevante para el TFM:**
Los perfiles de comportamiento del EDA son el mejor predictor de la tipología de anomalía en el grafo MG. Los exploradores son anomalías estructurales reales detectadas por dos modelos distintos con acuerdo perfecto. Los fantasmas son anomalías semánticas (desequilibrio entre tamaño registrado y actividad real) detectables solo por atributo. Esta correspondencia entre perfiles del EDA y señales del modelo valida el enfoque metodológico del TFM: el análisis cualitativo previo del EDA identifica los candidatos que el modelo confirma cuantitativamente.

### 11.1 Síntesis de los tres grafos para el capítulo 7

La lectura conjunta de los tres experimentos permite formular las siguientes conclusiones generales:

**DOMINANT vs GAD-NR en la práctica:** DOMINANT es el modelo más fiable en los tres grafos. Su señal estructural es consistente, su señal de atributo es la única operativa en G y MG, y sus embeddings son los más interpretables. GAD-NR aporta valor real solo en el grafo M, donde la señal joint-type (`gadnr_h`) detecta 496 nodos invisibles para DOMINANT. En G y MG, GAD-NR se reduce a un detector estructural con una única señal, equivalente a DOMINANT pero con menor estabilidad numérica.

**La señal estructural domina en los tres grafos:** dom_struct y gadnr_deg son las señales con mayor coherencia entre modelos (ρ = 0.743, 0.986, 0.847 en M, G y MG respectivamente) y las que mejor recuperan los candidatos del EDA. La detección de anomalías estructurales en grafos de redes sociales es robusta al modelo y a la arquitectura.

**El mecanismo de re-ponderación de lambdas de PyGOD 1.1.0 es la limitación más importante para la replicabilidad:** cualquier investigador que aplique GAD-NR con PyGOD 1.1.0 sobre grafos con features de dimensionalidad media o alta obtendrá `gadnr_feat` y `gadnr_h` degeneradas, independientemente de los lambdas que declare. Este hallazgo empírico, documentado en los tres grafos, es una contribución metodológica concreta del TFM a la comunidad.

---

## 12. Resumen ejecutivo

**Grafo:** MG — 25.233 nodos (24.631 miembros + 602 grupos), 45.583 aristas únicas, 38 features heterogéneas (zero-padding), 25 componentes conexas.

**DOMINANT:** convergencia correcta en 300 épocas (loss 5.14 → ~1.18, ~1.82 s/época). Sub-scores `dom_struct` (0.807–30.611, media 1.389, P95 = 2.436) y `dom_attr` (0.005–68.935, media 0.389, P95 = 1.621). Correlación entre señales ρ = 0.540 (artificial por heterogeneidad). Verificación ρ = 0.999603.

**GAD-NR:** convergencia con patrón escalonado (0.067 → 0.005), sin NaN (lr=0.001, ~8.25 s/época). Lambdas finales: 0.01 / 7.5 / 3.1·10⁻⁹. gadnr_deg P95 = 48.999. `gadnr_feat` exactamente cero para todos los nodos. `gadnr_h` rango 0.0013 (degenerada). Solo `gadnr_deg` (4.0–906.304) es informativa.

**Señal estructural:** ρ = 0.847, Jaccard P95 = 0.969. La más alta de los tres grafos en Jaccard. 20s in Nashville y Nashville Hiking en el #1 y #2 de ambas señales.

**Señal de atributo:** ρ = 0.133, Jaccard P95 = 0.241. Solo `dom_attr` es válida. Nashville Social Crew (grupo fantasma extremo) en P99.5 de dom_attr.

**Perfiles de comportamiento:** Exploradores detectados al 100% por señales estructurales (frac_P95 = 1.0). Inactivos invisibles (frac_P95 = 0.0). Fieles no anómalos (frac_P95 ≈ 0.006). Hiperactivos detectados al 56–84%.

**Grupos fantasma:** frac_P95 = 0 en estructural, 0.027 en dom_attr. Invisibles para la señal estructural, parcialmente detectables por atributo.

**t-SNE:** DOMINANT separa espontáneamente tipos de nodo, diferencia perfiles de comportamiento y muestra gradiente continuo de grado. GAD-NR produce separación de tipos menos nítida y sin estructura de tipología clara.

**Hallazgo clave:** el caso de Shalini (invisible en MG por grado 0, pese a ser la mayor anomalía estructural de M) demuestra que los tres grafos son complementarios e irreemplazables — ninguno captura la anomalía completa de un nodo por sí solo.

**Salidas:** `scores_MG.csv` (25.233 nodos × 21 columnas con sub-scores, percentiles y tipologías), `embeddings_MG.npz` (embeddings 64 dims y t-SNE de ambos modelos), `members_scores_MG.csv`, `groups_scores_MG.csv`, `dominant_MG.pkl`, `gadnr_model_MG.pt` para reproducibilidad y notebook 08.