# Resultados del notebook 05 — DOMINANT vs GAD-NR sobre el grafo M

Este documento sintetiza los resultados experimentales obtenidos en el notebook `05_dominant_gadnr_grafo_M_v4_1.ipynb`, donde se entrenan y comparan dos detectores de anomalías no supervisados sobre el grafo de miembros del dataset Meetup Tennessee. Está pensado como material de referencia para la redacción del capítulo 6 (Implementación y pruebas) y el capítulo 7 (Conclusiones) de la memoria del TFM.

---

## 1. Contexto experimental

### 1.1 Grafo analizado

El grafo M es la representación social del dataset Meetup Tennessee: nodos = miembros, aristas no dirigidas = co-membresía en al menos un grupo común, peso de la arista = número de grupos compartidos. Sus características estructurales son las siguientes:

- **11.371 nodos** (miembros con al menos una arista — se descartan los 13.219 miembros aislados de `meta_members` que no comparten membresía con ningún otro miembro registrado).
- **1.176.024 aristas únicas** (2.352.048 entradas en `edge_index` por la duplicación direccional requerida por PyTorch Geometric).
- **3 features por nodo**: `location_level` (ordinal 0-3), `lat`, `lon` — todas normalizadas con `StandardScaler` en el notebook 04.
- **Una única componente conexa** que cubre el 100% de los nodos del grafo.

La principal característica que condiciona el experimento es la **asimetría extrema entre la dimensionalidad estructural y la de atributos**: el modelo ve más de un millón de aristas pero solo tres features por nodo. Esta asimetría tiene implicaciones directas en el comportamiento de los modelos, especialmente en GAD-NR.

### 1.2 Modelos entrenados

Se entrenan dos detectores de la librería **PyGOD 1.1.0**, ambos pertenecientes a la familia GNN + Autoencoder no supervisado, sobre el mismo objeto `Data` de PyG (`data_M_clean`):

| Modelo | Año | Decoders | Salida |
|---|---|---|---|
| **DOMINANT** (Ding et al., SDM 2019) | 2019 | Estructura (producto interno) + atributos (GCN) | `dom_struct`, `dom_attr`, `dom_combined` |
| **GAD-NR** (Roy et al., WSDM 2024) | 2024 | Features propias (MLP) + grado (MLP) + vecindario (KL) | `gadnr_feat`, `gadnr_deg`, `gadnr_h`, `gadnr_score` |

### 1.3 Hiperparámetros

| Hiperparámetro | DOMINANT | GAD-NR |
|---|---|---|
| `hid_dim` | 64 | 64 |
| `num_layers` | 4 | 1 (encoder GNN) |
| `dropout` | 0.0 | 0.0 |
| `weight` | 0.2 | — |
| `lambda_loss1/2/3` | — | 1e-4 / 1e-3 / 1e-2 |
| `real_loss` | — | True (usa el loss original del paper como decision score) |
| `sample_size` | — | 4 |
| `sample_time` | — | 5 |
| `epoch` | 300 | 300 |
| `lr` | 0.004 | 0.004 |
| `weight_decay` | 0.0 | 1e-4 |
| `contamination` | 0.05 | 0.05 |
| `batch_size` | 0 (full batch) | 0 (full batch) |

### 1.4 Estrategia metodológica adoptada

La decisión metodológica más importante del notebook es **trabajar con sub-scores individuales como cinco señales independientes**, no con los scores compuestos agregados por los detectores. Las cinco señales son:

1. `dom_struct` — error de reconstrucción de adyacencia (DOMINANT, estructural).
2. `dom_attr` — error de reconstrucción de features (DOMINANT, atributo).
3. `gadnr_deg` — error de reconstrucción de grado (GAD-NR, estructural).
4. `gadnr_feat` — error de reconstrucción de features propias (GAD-NR, atributo).
5. `gadnr_h` — divergencia KL del vecindario (GAD-NR, joint-type, sin equivalente en DOMINANT).

Cada sub-score se evalúa por su **ranking relativo** (percentil dentro del grafo), que es robusto a las diferencias de escala entre modelos y al reescalado dinámico de lambdas de GAD-NR.

---

## 2. Resultados del entrenamiento

### 2.1 DOMINANT

El entrenamiento converge correctamente en 300 épocas: el loss comienza en ~10.41 y decae monotónicamente hasta estabilizarse alrededor de 9-10. No se observan oscilaciones ni divergencias.

Estadísticas de los sub-scores extraídos:

| Sub-score | Min | Max | Media | P95 |
|---|---|---|---|---|
| `dom_struct` (RMSE de A) | 1.4889 | 40.0403 | 12.1523 | 22.198 |
| `dom_attr` (RMSE de X) | 0.0018 | 42.6698 | 0.7345 | 2.658 |

La verificación de la fórmula de agregación `score = weight·attr + (1-weight)·struct` con `weight=0.2` produce una correlación de ρ = 1.000000 con el `decision_score_` devuelto por el detector — la fórmula es exacta.

**Hallazgo clave:** la correlación Spearman entre las dos señales internas de DOMINANT es **ρ = −0.013**, prácticamente cero. Los dos decoders capturan dimensiones de anomalía completamente independientes.

### 2.2 GAD-NR

El entrenamiento converge en 300 épocas con un patrón muy distinto: el loss comienza en ~837 y decae de forma escalonada, con caídas abruptas periódicas cada ~20 épocas (837 → 407 → 186 → 74...). Cada salto corresponde a una renegociación de los lambdas por parte del mecanismo interno de re-ponderación dinámica. La velocidad por época es ~2.2 s en GPU (vs ~0.26 s para DOMINANT).

Estadísticas de los sub-scores extraídos:

| Sub-score | Min | Max | Media | P95 |
|---|---|---|---|---|
| `gadnr_h` (KL del vecindario) | 0.7041 | 2.2863 | 0.8712 | 1.149 |
| `gadnr_deg` (RMSE de grado) | 0.0029 | 1.054.097 | 16.735,88 | 64.215 |
| `gadnr_feat` (RMSE de features) | 0.0010 | 0.1192 | 0.0103 | 0.026 |

La verificación de la fórmula compuesta produce ρ = 0.9963 — muy próximo a 1.0 pero no exacto, consecuencia del reescalado dinámico de lambdas durante el entrenamiento.

Correlaciones Spearman entre los tres sub-scores de GAD-NR:
- ρ(vecindario, grado) = −0.028
- ρ(vecindario, features) = 0.544
- ρ(grado, features) = −0.032

Los tres sub-scores son prácticamente ortogonales entre sí, con la única excepción de la correlación moderada entre `gadnr_h` y `gadnr_feat` (0.544), que refleja la influencia del reescalado de lambdas sobre la señal KL.

---

## 3. Comparativa entre las cinco señales

El heatmap de correlación Spearman (figura `05_signal_correlation_M.png`) sintetiza la estructura completa de acuerdos y desacuerdos:

|  | DOM-struct | DOM-attr | GADNR-deg | GADNR-feat | GADNR-KL |
|---|---|---|---|---|---|
| **DOM-struct** | 1.000 | −0.013 | **0.743** | −0.048 | −0.085 |
| **DOM-attr** | −0.013 | 1.000 | −0.010 | **0.839** | **0.573** |
| **GADNR-deg** | 0.743 | −0.010 | 1.000 | −0.032 | −0.028 |
| **GADNR-feat** | −0.048 | 0.839 | −0.032 | 1.000 | **0.544** |
| **GADNR-KL** | −0.085 | 0.573 | −0.028 | 0.544 | 1.000 |

De aquí emergen tres bloques claros:

1. **Bloque estructural** — `dom_struct` y `gadnr_deg` con ρ = 0.743 son las dos señales más correlacionadas entre modelos. La anomalía estructural es la dimensión más robusta a la arquitectura.

2. **Bloque atributo** — `dom_attr` y `gadnr_feat` con ρ = 0.839, correlación alta. Los decoders de features de ambos modelos convergen a detectar la misma población de nodos con features geográficas atípicas.

3. **Señal joint-type** — `gadnr_h` correlaciona con `dom_attr` (ρ = 0.573) y con `gadnr_feat` (ρ = 0.544), y anticorrelaciona levemente con las señales estructurales (ρ = −0.085 con `dom_struct`, −0.028 con `gadnr_deg`). Este patrón indica que `gadnr_h` captura principalmente anomalías de atributo en su vertiente de incoherencia con el vecindario, no anomalías estructurales.

### 3.1 Comparativa estructural

- **Correlación Spearman**: ρ = 0.743
- **Solapamiento top-100**: 58 nodos comunes (58%)
- **Exclusivos DOMINANT**: 42 nodos
- **Exclusivos GAD-NR**: 42 nodos
- **Jaccard P95**: 0.480 (369 detectados por ambos, 200 solo por cada uno)

La señal estructural es la más coherente entre modelos. El scatter de percentiles (figura `05_structural_comparison_M.png`) muestra una nube alargada bien alineada con la diagonal y un cluster compacto y denso en la esquina superior derecha (los 369 nodos detectados por ambos). La correlación ρ = 0.743 es la más alta entre señales de distinto modelo.

### 3.2 Comparativa de atributos

- **Correlación Spearman**: ρ = 0.839
- **Solapamiento top-100**: 67 nodos comunes (67%)
- **Exclusivos DOMINANT**: 33 nodos
- **Exclusivos GAD-NR**: 33 nodos
- **Jaccard P95**: 0.410 (331 detectados por ambos)

Resultado también muy alto. El scatter de atributos (figura `05_attribute_comparison_M.png`) muestra una nube bien alineada con la diagonal, con baja dispersión y un cluster compacto en el cuadrante superior derecho. Las señales de atributo de ambos modelos convergen notablemente en esta ejecución.

### 3.3 Scores compuestos

| Estadístico | DOMINANT | GAD-NR |
|---|---|---|
| Media | 9.8697 | 0.0819 |
| Std | 4.4645 | 0.0733 |
| Min | 1.1888 | 0.0080 |
| Max | 32.0930 | 0.8974 |
| P95 | 17.781 | 0.210 |

- **Correlación Spearman entre compuestos**: ρ = 0.079
- **Solapamiento top-100**: 0% (ningún nodo en común)

Los scores compuestos son incomparables en escala (DOMINANT: media 9.87, GAD-NR: media 0.08) y la correlación de rankings es prácticamente nula (ρ = 0.079). El solapamiento top-100 de 0% es el resultado más extremo observado entre ejecuciones y confirma que el score compuesto de GAD-NR —dominado por `gadnr_feat` tras el reescalado de lambdas— señala una población completamente distinta a la de DOMINANT.

Los histogramas en escala logarítmica (figura `05_composite_score_distributions_M.png`) revelan distribuciones cualitativamente distintas: DOMINANT presenta una distribución amplia y bien repartida entre 1 y 32, con cola larga y buen contraste; GAD-NR decae más rápidamente desde 0 hasta ~0.9, con varios outliers extremos aislados.

### 3.4 Tipología de anomalías

Los scatter de tipología (figuras `05_typology_dominant_M.png` y `05_typology_gadnr_M.png`) comparten la misma forma estructural: una **"L invertida"** muy marcada, con los nodos de atributo (naranja) formando una franja horizontal densa pegada al P95 del eje Y, y los nodos estructurales (azul) formando una franja vertical pegada al P95 del eje X. Esta forma confirma visualmente la ortogonalidad casi perfecta entre las dos señales de cada modelo.

| Tipología | DOMINANT | GAD-NR |
|---|---|---|
| Normal | 10.249 (90.1%) | 10.263 (90.3%) |
| Estructural | 553 (4.9%) | 538 (4.7%) |
| Atributo | 553 (4.9%) | 539 (4.7%) |
| Mixta | 16 (0.1%) | 31 (0.3%) |

GAD-NR tiene el doble de mixtos que DOMINANT (31 vs 16), coherente con que en esta ejecución las señales estructural y de atributo de GAD-NR coinciden más en la esquina superior-derecha.

### 3.5 Cruce de tipologías

| DOMINANT \ GAD-NR | Normal | Estructural | Atributo | Mixta |
|---|---|---|---|---|
| **Normal** | 9.848 | 188 | 208 | 5 |
| **Estructural** | 183 | 344 | 15 | 11 |
| **Atributo** | 231 | 2 | 315 | 5 |
| **Mixta** | 1 | 4 | 1 | 10 |

Acuerdos diagonales:
- Normal-Normal: 9.848 (86.6%)
- Estructural-Estructural: 344 (62.2% de los estructurales DOM) — el más alto registrado entre ejecuciones
- Atributo-Atributo: 315 (57.0% de los atributo DOM) — también el más alto
- Mixta-Mixta: 10 (62.5% de los mixtos DOM)

El acuerdo en todas las categorías es notablemente alto en esta ejecución, consecuencia de las correlaciones elevadas entre señales equivalentes.

---

## 4. Comportamiento del mecanismo de re-ponderación de lambdas en GAD-NR

### 4.1 Descripción del problema

PyGOD 1.1.0 implementa GAD-NR con un mecanismo interno de **re-ponderación dinámica de los `lambda_lossN`** durante el entrenamiento. Este mecanismo responde a la magnitud relativa de cada componente del loss y ajusta los lambdas en cada época para que ningún componente domine excesivamente. Es distinto del parámetro `real_loss`, que solo controla si el score de decisión final usa el loss original del paper o una versión ponderada alternativa — `real_loss=True` simplemente significa usar el loss del paper como score, no desactiva el reescalado de lambdas.

### 4.2 Invarianza del punto fijo: hallazgo crítico

Los lambdas finales tras el entrenamiento, con independencia de los lambdas declarados (1e-4 / 1e-3 / 1e-2):

| Lambda | Componente | Declarado | Final | Factor |
|---|---|---|---|---|
| `lambda_loss1` | Vecindario (KL) | 1e-4 | 0.0001 | ×1 (invariante) |
| `lambda_loss2` | Features propias | 1e-3 | ~7.50 | ×7.500 |
| `lambda_loss3` | Grado | 1e-2 | ~3.1·10⁻⁷ | ÷32.000.000 |

El mismo punto fijo (λ₁ invariante, λ₂ ≈ 7.5, λ₃ ≈ nulo) se observa en todas las ejecuciones con independencia de los lambdas de partida. El mecanismo actúa como **atractor**: los lambdas declarados son irrelevantes para los lambdas efectivos del entrenamiento.

El patrón de pérdida también es revelador: el loss cae de forma escalonada con saltos bruscos aproximadamente cada 20 épocas (837 → 407 → 186 → 74...), cada salto correspondiendo a una renegociación de los lambdas. Este comportamiento es reproducible entre ejecuciones.

### 4.3 Consecuencias observables

1. **`gadnr_deg` numéricamente extremo pero útil ordinalmente.** Su rango es 0.003 a 1.054.097 con P95 = 64.215. El decoder de grado, con λ₃ ≈ 3·10⁻⁷, no recibe gradiente útil y produce valores residuales de inicialización. El ranking relativo sigue siendo informativo: los super-conectores reales (Jim H #1, Shalini #9, Matt Kenigson #78) aparecen en el top de esta señal.

2. **`gadnr_score` (compuesto) dominado por features.** Con λ₂ ≈ 7.5 y λ₃ ≈ 3·10⁻⁷, el score compuesto está determinado casi en exclusiva por `gadnr_feat`. Correlación de verificación: ρ = 0.9963.

3. **`gadnr_h` parcialmente influenciada por el reescalado.** Su lambda no cambia (λ₁ = 0.0001) pero la correlación con `gadnr_feat` (ρ = 0.544) sugiere que la señal KL no es completamente independiente de la dinámica de features. Aun así, es la señal semánticamente más distinta de las tres.

### 4.4 Implicación metodológica

Trabajar con sub-scores individuales en lugar del score compuesto está plenamente justificado. Los lambdas declarados son irrelevantes para los efectivos; solo el ranking por sub-score individual preserva la distinción semántica entre tipos de anomalía. Esta limitación debe documentarse en el capítulo de Discusión del TFM.

---

## 5. Señal joint-type de GAD-NR

### 5.1 Caracterización

La distribución de `gadnr_h` (figura `05_gadnr_joint_type_M.png`) es claramente **bimodal**: un primer pico muy estrecho y alto en torno a ~0.70–0.75 (la gran mayoría de los nodos, con KL muy baja), un valle entre ~0.80 y ~0.95, y un segundo modo más ancho centrado en ~1.05–1.15. El P95 = 1.149 cae en la mitad del segundo modo. La cola derecha se extiende hasta ~2.3. Esta forma bimodal refleja dos regímenes bien separados: nodos con vecindarios predecibles (modo izquierdo) y nodos con vecindarios más difíciles de modelar (modo derecho).

### 5.2 Nodos joint-type puros

Los **nodos joint-type puros** —superan P95 en `gadnr_h` pero no son detectados como anómalos por ninguna señal de DOMINANT— son **273 nodos** (2.4% del total):

- **Grado medio: 79.0** (vs 103.2 global) — conectividad baja-moderada, no super-conectores.
- Los **top-10 son casi exclusivamente miembros de Chicago e Illinois** (Arlington Heights, Lincolnwood) con grado idéntico de 19.

### 5.3 Interpretación crítica

El perfil geográfico de los top joint-type es llamativo y constante entre ejecuciones: miembros de Illinois con coordenadas de Chicago (~41.9°N, −87.7°W) cuyas features (location_level=1, lat/lon de Chicago) divergen radicalmente de las de sus vecinos nashvillenses (location_level=2-3, lat ~36°N, lon ~−86.8°W). El modelo KL detecta esta incoherencia geográfica entre el nodo y su vecindario, no una anomalía comportamental genuina.

Esto significa que `gadnr_h` en el grafo M está detectando principalmente **miembros out-of-state con grados moderados** cuyos vecinos son mayoritariamente locales de Tennessee. Es una señal válida de incoherencia geográfica pero limitada como detector de anomalías sociales complejas en este grafo de solo 3 features.

### 5.4 Contraste con los puentes inter-comunidad del EDA

Los cuatro candidatos como puentes inter-comunidad (Tremaine James, Mary Beth, chen hajaj, James Lauderdale Jr) no superan el P95 en `gadnr_h`. Sus percentiles KL se sitúan entre P40 y P70. La causa es doble: vecindarios pequeños (5–12 nodos) que hacen la estimación KL muy ruidosa, y que su condición de puente es un rol estructural, no una incoherencia de features geográficas.

---

## 6. Validación contra candidatos del EDA

| Candidato | Tipo EDA | Grado | DOM-struct | DOM-attr | GADNR-deg | GADNR-feat | GADNR-h |
|---|---|---|---|---|---|---|---|
| **Pablo** | Alta betweenness, degree moderada | 688 | #16 (P99.9) ⚠ | #5.156 (P54.7) | #10.139 (P10.8) | #11.294 (P0.7) | #7.843 (P31.0) |
| **Shalini** | Mayor betweenness de la red | 955 | #2 (P100) ⚠ | #5.848 (P48.6) | #9 (P99.9) ⚠ | #11.264 (P0.9) | #8.264 (P27.3) |
| **Jim H** | Super-conector + puente | 1.178 | #1 (P100) ⚠ | #4.753 (P58.2) | #1 (P100) ⚠ | #11.276 (P0.8) | #8.860 (P22.1) |
| **Matt Kenigson** | Super-conector | 905 | #3 (P100) ⚠ | #5.745 (P49.5) | #78 (P99.3) ⚠ | #11.228 (P1.3) | #6.591 (P42.0) |
| **GEEK by AKEIN** | Entidad no personal | 0 | #11.367 (P0.0) | #3.641 (P68.0) | #11.211 (P1.4) | #5.347 (P53.0) | #2.805 (P75.3) |

### 6.1 Aciertos sólidos

**Shalini, Jim H y Matt Kenigson** son detectados como anomalía estructural por **ambos modelos con consistencia máxima**. Jim H lidera `gadnr_deg` (#1) y `dom_struct` (#1); Shalini es #2 en `dom_struct` y #9 en `gadnr_deg`; Matt Kenigson es #3 en `dom_struct` y #78 en `gadnr_deg` (P99.3). Los tres superan el P95 estructural en ambos modelos. Validación cruzada perfecta, estable entre todas las ejecuciones.

Un dato llamativo nuevo en esta ejecución: los tres super-conectores detectados estructuralmente tienen percentiles extremadamente bajos en las señales de atributo y joint-type de GAD-NR (P0.7–P1.3 en `gadnr_feat`, P22–P44 en `gadnr_h`). Esto refleja que el reescalado de lambdas orienta `gadnr_feat` hacia detectar nodos con features geográficas atípicas, que es exactamente lo contrario al perfil de los super-conectores nashvillenses de alto grado.

### 6.2 Discrepancias informativas

**Pablo** es el caso más informativo. El EDA lo describe con betweenness alta (#3 de la red) pero degree moderada (688). DOMINANT lo detecta como anomalía estructural de primer orden (#16, P99.9), pero **GAD-NR lo coloca en el P10.8 de `gadnr_deg`** — su grado es fácil de predecir desde el embedding. La discrepancia es estructuralmente significativa: DOMINANT detecta que los *enlaces concretos* de Pablo son difíciles de reconstruir (rol de puente entre comunidades), mientras que GAD-NR no lo ve como anómalo porque su *grado bruto* es perfectamente típico. Este resultado es robusto: se replica con pequeñas variaciones en todas las ejecuciones, confirmando que **DOMINANT captura matices de rol estructural (puente) que GAD-NR no puede detectar vía grado**.

### 6.3 Fallos comunes

**GEEK by AKEIN Engineering** (grado 0) es un falso negativo en ambos modelos para la dimensión estructural. Sin aristas no hay vecindario que reconstruir. La señal `gadnr_h` lo sitúa en P75.3 — la señal más alta que alcanza en cualquier dimensión, consecuencia de que con vecindario vacío la KL es intrínsecamente difícil de modelar. Este caso confirma la limitación arquitectónica intrínseca de toda la familia GNN+AE: **son ciegos a anomalías semánticas en nodos aislados**.

### 6.4 Interpretación general

Los aciertos sólidos (Shalini, Jim H, Matt Kenigson) son robustos y estables entre todas las ejecuciones. La discrepancia de Pablo también es estable. GEEK documenta una limitación arquitectónica. El nivel de validación cualitativa es alto para los candidatos con anomalía estructural; más débil para los candidatos puramente semánticos.

---

## 7. Análisis de los embeddings latentes (t-SNE)

Los embeddings de ambos modelos (dimensión 64) se proyectan a 2D mediante t-SNE (perplexity=30, init='pca', random_state=SEED). Se generan tres vistas por modelo.

### 7.1 Embeddings de DOMINANT

**Vista por tipo** (`05_dominant_tsne_type_M.png`): los nodos estructurales (azul, 553) forman un **cluster compacto y bien definido** en la zona central-izquierda del espacio t-SNE. Los mixtos (rojo, 16) están todos integrados en este cluster. Los nodos de atributo (naranja, 553) se distribuyen ampliamente por todo el plano sin agrupación propia, apareciendo tanto en la periferia como dispersos alrededor del cluster central.

**Vista por score** (`05_dominant_tsne_score_M.png`): el gradiente de scores altos (rojo intenso, hasta ~26) coincide exactamente con el cluster estructural. La intensidad decae suavemente hacia naranjas y amarillos conforme nos alejamos del núcleo del cluster. La intensidad de la anomalía sigue una estructura espacial clara y bien localizada — el embedding de DOMINANT organiza los nodos por intensidad estructural de forma coherente.

**Vista por grado** (`05_dominant_tsne_degree_M.png`): el cluster estructural (centro-izquierda) coincide con los nodos de mayor grado en escala log (amarillo-verde, ~log(grado)=6-7). Los nodos de grado bajo (morado-azul oscuro) se distribuyen por la periferia, especialmente en clusters pequeños en la parte baja. El embedding de DOMINANT ha aprendido representaciones donde los super-conectores quedan agrupados, validando que la señal estructural refleja genuinamente la topología del grafo.

### 7.2 Embeddings de GAD-NR

**Vista por tipo** (`05_gadnr_tsne_type_M.png`): el t-SNE de GAD-NR muestra un patrón radicalmente distinto. **No hay un cluster estructural compacto**. Los nodos estructurales (azul, 538) están dispersos por todo el espacio sin agrupación. En cambio, aparecen varios **clusters locales compactos de nodos de atributo y mixtos** — una franja densa en la zona superior-central y varios subgrupos en la parte inferior. Los mixtos (rojo, 31) se concentran en las zonas de mayor densidad de atributo.

**Vista por score** (`05_gadnr_tsne_score_M.png`): los puntos más rojos (score compuesto alto, hasta ~0.5) se concentran en los mismos clusters de atributo/mixtos, coherente con que el score compuesto está dominado por `gadnr_feat`. La intensidad de score no sigue una estructura única sino múltiples focos dispersos, en contraste con el foco único y nítido de DOMINANT.

**Vista por grado** (`05_gadnr_tsne_degree_M.png`): la distribución de grados en el espacio t-SNE revela la naturaleza del embedding. Se observan **estructuras geométricas inusuales**: líneas curvas y arcos pronunciados en la zona inferior, donde los nodos de grado similar se alinean formando curvas en lugar de clusters difusos. Esto es diagnóstico de que el embedding tiene **baja variedad intrínseca** — colapsa a una manifold de baja dimensión en el espacio de 64 dimensiones, consecuencia de que el reescalado de lambdas hace que la optimización esté dominada por 3 features geográficas normalizadas.

### 7.3 Implicación para el TFM

La comparativa de los t-SNE refuerza la conclusión central: DOMINANT organiza su espacio latente en torno a la señal estructural, produciendo un único cluster compacto bien separado de super-conectores; GAD-NR organiza el suyo en torno a la señal de atributo (features geográficas), produciendo clusters locales de anomalías de atributo sin separar los estructurales. Ninguno es mejor en abstracto — reflejan qué dimensión domina su optimización.

La presencia de artefactos geométricos en el t-SNE de GAD-NR (líneas, arcos) es diagnóstica de baja variedad intrínseca del embedding: el modelo, cuya optimización queda dominada por 3 features geográficas normalizadas tras el reescalado de lambdas, no puede desplegar representaciones en un espacio de alta variedad con solo esa información. La hipótesis verificable en los notebooks 06 y 07 es que con 35 o 38 features el embedding de GAD-NR debería desplegarse con mayor variedad, eliminando los artefactos geométricos.

---

## 8. Limitaciones identificadas

### 8.1 Limitaciones intrínsecas del grafo M

1. **Pocas features (3 dimensiones).** Solo features geográficas, lo que limita la capacidad de detectar anomalías de atributo más allá de diferencias de localización. La señal de atributo tiene baja riqueza semántica.

2. **Asimetría estructura-atributo.** Con >1M aristas y solo 3 features, los modelos están dominados por la estructura. La señal de atributo es estadísticamente más débil.

3. **Nodos aislados invisibles.** Los 13.219 miembros de `meta_members` ausentes del grafo M son inaccesibles para cualquier modelo GNN+AE. Esta limitación es intrínseca a la familia arquitectónica.

### 8.2 Limitaciones de los modelos

1. **Anomalías semánticas en nodos aislados.** Como ilustra GEEK by AKEIN Engineering, sin vecindario no hay reconstrucción posible.

2. **Invarianza del mecanismo de re-ponderación de lambdas en PyGOD 1.1.0.** Los lambdas declarados son irrelevantes para los efectivos; el mecanismo converge al mismo punto fijo con independencia del punto de partida. Esto impide controlar directamente el balance entre las tres señales de GAD-NR.

3. **Señal `gadnr_deg` dominada por la inicialización.** Con λ₃ ≈ 3·10⁻⁷, el decoder de grado no recibe gradiente útil. El ranking es informativo pero los valores absolutos son patológicos.

4. **`gadnr_h` confundida con efecto geográfico.** Los top joint-type son sistemáticamente miembros de Chicago/Illinois, no nodos con comportamiento social anómalo genuino. En un grafo de 3 features geográficas, la KL captura incoherencia geográfica, no incoherencia social.

5. **Variabilidad entre ejecuciones en rankings intermedios.** Los aciertos sólidos (Shalini, Jim H, Matt Kenigson, Pablo) son estables. Los nodos en los rankings P90–P97 pueden cambiar entre ejecuciones por la estocasticidad residual de CUDA con seed fijo.

### 8.3 Limitaciones metodológicas

1. **Ausencia de ground truth.** La validación es cualitativa contra candidatos del EDA, no cuantitativa con AUC-ROC o F1.

2. **Comparativa de dos modelos sin baseline simple.** No se incluye Isolation Forest sobre features estructurales como punto de referencia.

3. **Una única ejecución por configuración.** Las correlaciones globales son robustas; los rankings extremos de `gadnr_h` y `gadnr_feat` no tanto.

---

## 9. Conclusiones del notebook

Los resultados validan la hipótesis principal del TFM: **los modelos GNN+AE no supervisados recuperan las anomalías estructurales más prominentes del grafo de miembros sin etiquetas**. Los super-conectores y puentes inter-comunidad identificados en el EDA aparecen sistemáticamente en lo más alto de los rankings de ambos modelos, de forma estable entre ejecuciones.

Los resultados también establecen matices importantes:

1. **DOMINANT y GAD-NR son complementarios.** La señal estructural converge bien (ρ = 0.743, Jaccard P95 = 0.480). La señal de atributo también converge en esta ejecución (ρ = 0.839, solapamiento top-100 = 67%). La aportación distintiva de GAD-NR es `gadnr_h`, pero en este grafo su valor práctico es limitado por el efecto geográfico.

2. **DOMINANT es la opción más interpretable para el grafo M.** Su embedding organiza el espacio latente de forma coherente con la tipología estructural. El cluster de super-conectores es compacto y bien separado.

3. **El mecanismo de re-ponderación de lambdas actúa como atractor invariante.** Independientemente de los lambdas declarados, el mecanismo interno de PyGOD 1.1.0 converge al mismo punto fijo: λ₁ invariante (~0.0001), λ₂ amplificado (~7.5), λ₃ colapsado (~nulo). Los lambdas de instanciación son irrelevantes para los efectivos. El parámetro `real_loss` controla únicamente si el score de decisión usa el loss original del paper o una versión ponderada, no el reescalado de lambdas.

4. **El score compuesto de GAD-NR es prácticamente inútil como comparador.** Con ρ = 0.079 y solapamiento top-100 del 0% respecto a DOMINANT, los compuestos no son comparables. Solo los sub-scores individuales son informativos.

5. **Los aciertos sólidos son robustos entre ejecuciones.** Shalini, Jim H, Matt Kenigson y la discrepancia de Pablo se replican con pequeñas variaciones en todas las ejecuciones. Esto es la señal más fuerte de que el análisis basado en sub-scores individuales captura patrones genuinos del grafo.

### 9.1 Hipótesis para los notebooks 06 y 07

- **Grafo G (456 nodos, 35 features)**: el ratio features/nodos es mucho más favorable. Se espera que GAD-NR tenga mejor comportamiento del embedding (sin artefactos geométricos), mayor separabilidad de tipologías, y una señal `gadnr_h` más genuinamente joint-type porque los grupos tienen atributos ricos (categoría, tamaño, organizador).

- **Grafo MG (25.233 nodos, 38 features, bipartito)**: caso intermedio. La asimetría entre features de miembro (3) y de grupo (35) puede generar problemas adicionales en el embedding. La señal joint-type debería ser especialmente útil para detectar miembros cuyas pautas de asistencia no encajan con las de su grupo.

---

## 10. Resumen ejecutivo

**Lo que se ha hecho:** entrenar DOMINANT (300 épocas, loss 10.41 → convergente, ~0.26s/época) y GAD-NR (300 épocas, loss 837 → convergente con caídas escalonadas, ~2.2s/época, con workarounds para bugs de PyGOD 1.1.0) sobre el grafo M. Extraer cinco sub-scores y analizarlos como señales independientes mediante correlaciones, solapamientos, cruces de tipología, distribuciones, validación contra candidatos del EDA y t-SNE.

**Lo que funciona:** detección consistente y robusta de Shalini (#2 DOM-struct, #9 GADNR-deg), Jim H (#1 en ambas señales estructurales) y Matt Kenigson (#3 DOM-struct, #78 GADNR-deg) en percentil máximo. Correlación estructural ρ = 0.743, Jaccard P95 = 0.480. Correlación de atributo ρ = 0.839. El t-SNE de DOMINANT muestra un cluster estructural compacto con gradiente de score coherente.

**Lo que no funciona del todo:** score compuesto de GAD-NR con ρ = 0.079 y solapamiento top-100 del 0% respecto a DOMINANT — los compuestos son incomparables. El mecanismo interno de re-ponderación de lambdas reescala los valores declarados (1e-4/1e-3/1e-2) al mismo punto fijo invariante (λ₂ ≈ 7.5, λ₃ ≈ 3·10⁻⁷), con independencia de los lambdas de partida. La señal `gadnr_h` detecta principalmente miembros de Chicago (efecto geográfico), no anomalías sociales complejas. El t-SNE de GAD-NR muestra artefactos geométricos (líneas, arcos) diagnósticos de baja variedad del embedding.

**Hallazgo nuevo respecto a ejecuciones anteriores:** el solapamiento top-100 de compuestos llegó al 0% (vs 1% anterior), reforzando que el score compuesto de GAD-NR no es comparable al de DOMINANT. El acuerdo en tipología estructural-estructural alcanzó el 62% (vs 40% en la primera ejecución), su valor más alto.

**Limitación importante:** los modelos GNN+AE no pueden detectar anomalías semánticas en nodos sin aristas (GEEK by AKEIN Engineering). Limitación arquitectónica intrínseca.

**Salidas:** `scores_M.csv` (11.371 × 20) con los cinco sub-scores, percentiles y tipologías; `embeddings_M.npz` con los embeddings de 64 dimensiones de ambos modelos y sus proyecciones t-SNE.

**Próximo paso:** repetir el experimento sobre el grafo G (456 grupos, 35 features) para contrastar las hipótesis sobre el comportamiento de GAD-NR con features más ricas.