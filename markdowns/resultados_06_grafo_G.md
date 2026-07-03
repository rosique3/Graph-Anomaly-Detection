# Resultados del notebook 06 — DOMINANT vs GAD-NR sobre el grafo G

Este documento sintetiza los resultados experimentales obtenidos en el notebook `06_dominant_gadnr_grafo_G.ipynb`, donde se entrenan y comparan dos detectores de anomalías no supervisados sobre el grafo de grupos del dataset Meetup Tennessee. Está pensado como material de referencia para la redacción del capítulo 6 (Implementación y pruebas) y el capítulo 7 (Conclusiones) de la memoria del TFM, en combinación con el documento equivalente del grafo M (`resultados_05_grafo_M.md`).

---

## 1. Contexto experimental

### 1.1 Grafo analizado

El grafo G es la representación de la red de grupos del dataset Meetup Tennessee: nodos = grupos, aristas no dirigidas = solapamiento de al menos un miembro común, peso = número de miembros compartidos. Sus características estructurales son:

- **456 nodos** (grupos con al menos una arista — los 146 grupos sin co-membresía con ningún otro quedan excluidos de los 602 totales de `meta_groups`).
- **6.692 aristas únicas** (13.384 entradas en `edge_index` por duplicación direccional).
- **35 features por nodo**: `log_num_members`, `log_num_events`, `is_truncated`, `has_valid_organizer` y one-hot de `category_name` (31 categorías). Todas normalizadas en el notebook 04.
- **Densidad: 0.065** — casi cuatro veces superior a la del grafo M (0.018).
- **Grado medio: 29.35**, rango 0–182.
- **Una única componente conexa** que cubre el 100% de los nodos del grafo.

### 1.2 Diferencias clave respecto al grafo M

| Aspecto | Grafo M | Grafo G |
|---|---|---|
| Nodos | 11.371 miembros | 456 grupos |
| Aristas únicas | 1.176.024 | 6.692 |
| Features/nodo | 3 (geográficas) | 35 (categoría, tamaño, actividad) |
| Densidad | 0.018 (disperso) | 0.065 (×4 más denso) |
| Grado medio | 207 | 29 |
| Grado máximo | 2.356 | 182 |

Estas diferencias condicionan profundamente el comportamiento de los modelos y producen resultados cualitativamente distintos a los del grafo M, especialmente en lo relativo al curriculum de GAD-NR y a la calidad relativa de las señales de atributo.

### 1.3 Modelos y configuración

Se entrenan los mismos dos detectores que en el grafo M (DOMINANT y GAD-NR de PyGOD 1.1.0), con los mismos workarounds para bugs de PyGOD. Los cambios de hiperparámetros respecto a M están todos justificados:

| Hiperparámetro | Grafo M | Grafo G | Justificación |
|---|---|---|---|
| `hid_dim` | 64 | 64 | Sin cambio — misma capacidad de representación |
| `weight` (DOM) | 0.2 | 0.5 | Equilibrio features/estructura con 35 features ricas |
| `lr` (DOMINANT) | 0.004 | 0.004 | Sin cambio — entrenamiento estable |
| `lr` (GAD-NR) | 0.004 | 0.001 | Reducido para evitar divergencia NaN |
| `weight_decay` (GADNR) | 1e-4 | 1e-3 | Más regularización — grafo más pequeño |
| `dropout` | 0.0 | 0.0 | Sin cambio — coherencia metodológica |
| `lambda_lossN` | 0.01/0.1/0.8 | 0.01/0.1/0.8 | Sin cambio — comparabilidad entre grafos |
| `contamination` | 0.05 | 0.05 | Sin cambio — P95 como umbral de anomalía |

---

## 2. Resultados del entrenamiento

### 2.1 DOMINANT

El entrenamiento converge correctamente en 300 épocas sin ninguna inestabilidad. La verificación de la fórmula compuesta da ρ = 1.000000 — los sub-scores se extraen con precisión perfecta.

Estadísticas de los sub-scores:

| Sub-score | Min | Max | Media | P95 |
|---|---|---|---|---|
| `dom_struct` (RMSE de A) | 0.866 | 8.883 | 3.858 | 6.954 |
| `dom_attr` (RMSE de X) | 0.143 | 2.807 | 1.241 | 2.063 |
| `dom_combined` | 0.800 | 5.334 | 2.550 | 4.050 |

**Hallazgo clave:** la correlación Spearman entre las dos señales internas de DOMINANT es **ρ = −0.150** — ligeramente negativa y más pronunciada que en M (ρ = −0.004). Los grupos estructuralmente atípicos (alta conectividad) tienden a tener features normales (son representativos de su categoría), y los grupos con features atípicas tienden a tener pocas conexiones. La ortogonalidad entre señales es real y bien definida en G.

Las distribuciones de cada señal tienen formas distintas: `dom_struct` presenta asimetría positiva con una cola derecha bien separada (buen contraste entre normales y anómalos); `dom_attr` es más simétrica y compacta, sin la cola larga de `dom_struct`.

### 2.2 GAD-NR

El entrenamiento con `lr=0.004` divergió a NaN a partir de la época ~277 — el loss comenzó en 1529, descendió monotónicamente hasta ~−0.02 en la época 200, y a partir de la 277 produjo NaN en cascada que corrompieron todos los pesos del modelo. La causa es que con 35 features el loss de features parte de valores muy altos, y cuando se acerca a cero el optimizador produce gradientes inestables. Reducir `lr=0.001` y aumentar `weight_decay=1e-3` resolvió el problema — el entrenamiento es estable y la verificación da ρ = 1.000000.

Estadísticas de los sub-scores:

| Sub-score | Min | Max | Media | P95 |
|---|---|---|---|---|
| `gadnr_h` (KL vecindario) | 0.3521 | 0.3539 | 0.3531 | 0.353 |
| `gadnr_deg` (RMSE grado) | 3.470 | 33.439 | 1.951 | 10.224 |
| `gadnr_feat` (RMSE features) | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| `gadnr_score` | 0.004 | 0.820 | 0.051 | 0.253 |

Los lambdas finales tras curriculum son **idénticos a los del grafo M**: 0.01 / 7.6 / 2.4·10⁻⁵. Las correlaciones entre sub-scores de GAD-NR son todas cercanas a cero (máximo ρ = −0.100 entre grado y features), confirmando ortogonalidad.

---

## 3. Comportamiento del curriculum de GAD-NR en G

### 3.1 El hallazgo principal

Los lambdas finales tras el entrenamiento son **idénticos a los del grafo M** (0.01 / 7.6 / 2.4·10⁻⁵), a pesar de que los dos grafos tienen características muy distintas: número de nodos, densidad, rango de grados y dimensionalidad de features completamente diferentes. Esto contradice la hipótesis inicial — que en G el curriculum produciría un reescalado más equilibrado — y establece que **el mecanismo de re-ponderación de PyGOD converge siempre al mismo punto fijo independientemente del grafo**.

Este hallazgo tiene implicaciones importantes para el TFM: el curriculum no es un mecanismo adaptativo al grafo sino una propiedad fija de la implementación de PyGOD 1.1.0. Cualquier usuario que aplique GAD-NR con `real_loss=True` sobre cualquier grafo obtendrá los mismos lambdas finales, independientemente de sus lambdas iniciales.

### 3.2 La paradoja del grafo G

En M, con 3 features de baja varianza, el curriculum amplificó `lambda_loss2` ×76 porque la señal de features era débil y necesitaba amplificación para no ser ignorada. El resultado fue que `gadnr_feat` en M tenía varianza reducida (rango 0.04–0.24) pero aún informativa.

En G, con 35 features ricas y semánticamente significativas, el mismo reescalado (×76) hace que el decoder aprenda a reconstruirlas con error prácticamente cero para todos los grupos. El resultado es que `gadnr_feat` en G tiene varianza nula (rango 0.0000–0.0000) y es completamente inútil como señal de anomalía.

**La paradoja es total:** cuanto más informativas son las features, más agresivamente las sobreoptimiza el curriculum, anulando precisamente las señales que más información nueva aportarían. En G, DOMINANT — sin decoder de vecindario ni señal joint-type explícita — produce señales de atributo más informativas que GAD-NR.

### 3.3 Consecuencias en `gadnr_h`

La señal joint-type (`gadnr_h`) también queda degenerada en G: rango 0.3521–0.3539, variación total de 0.0018. La causa es la misma — el curriculum sobreoptimiza las features hasta el punto de que también las distribuciones de vecindario quedan perfectamente modeladas para todos los grupos sin excepción. En M, donde el curriculum era menos extremo en sus consecuencias sobre features, `gadnr_h` mantenía varianza suficiente (rango 0.94–3.55) para ser la señal más fiable.

### 3.4 Única señal operativa de GAD-NR en G

De las tres señales de GAD-NR, solo `gadnr_deg` es informativa en G. Su rango (3.47–33.439) es manejable gracias al menor rango de grados del grafo G (0–182 vs 0–2.356 en M), y su ranking relativo captura correctamente los grupos estructuralmente atípicos. El análisis posterior se apoya casi exclusivamente en esta señal.

---

## 4. Comparativa entre las cinco señales

El heatmap de correlación Spearman es el resultado más informativo del notebook:

|  | DOM-struct | DOM-attr | GADNR-deg | GADNR-feat | GADNR-KL |
|---|---|---|---|---|---|
| **DOM-struct** | 1.000 | −0.150 | **0.986** | −0.086 | −0.025 |
| **DOM-attr** | −0.150 | 1.000 | −0.201 | 0.061 | **−0.401** |
| **GADNR-deg** | 0.986 | −0.201 | 1.000 | −0.100 | −0.003 |
| **GADNR-feat** | −0.086 | 0.061 | −0.100 | 1.000 | −0.069 |
| **GADNR-KL** | −0.025 | −0.401 | −0.003 | −0.069 | 1.000 |

Los patrones más relevantes:

**Bloque estructural dominante:** `dom_struct` y `gadnr_deg` con ρ = 0.986 forman el bloque más coherente de todo el experimento — muy superior al ρ = 0.668 de M. Las dos señales estructurales son prácticamente equivalentes en G.

**Bloque de atributo inexistente:** `dom_attr` y `gadnr_feat` tienen ρ = 0.061 — prácticamente cero. No existe correlación entre las dos señales de atributo porque `gadnr_feat` es constante.

**`dom_attr` vs `gadnr_h`: ρ = −0.401** — correlación negativa moderada. Este es el hallazgo más sorprendente del heatmap. En M esta correlación era positiva (ρ = 0.589), lo que sugería que parte de lo que DOMINANT detecta como anomalía de atributo, GAD-NR lo recoge como divergencia de vecindario. En G la relación se invierte: los grupos con alto error de atributo en DOMINANT tienden a tener KL de vecindario baja en GAD-NR. La explicación es que con `gadnr_h` degenerada (rango 0.0018), cualquier correlación con otras señales refleja artefactos del curriculum más que relaciones genuinas entre tipos de anomalía.

**Correlaciones negativas entre señales estructurales y de atributo:** `dom_struct` vs `dom_attr` (−0.150) y `gadnr_deg` vs `gadnr_feat` (−0.100) y `gadnr_deg` vs `dom_attr` (−0.201) son todas negativas. En G los grupos más conectados tienden a tener features más normales, y viceversa — una separación entre dimensiones de anomalía más clara que en M.

### 4.1 Comparativa estructural

- **Correlación Spearman**: ρ = 0.986
- **Solapamiento top-50**: 37/50 (74%)
- **Outliers P95 ambos modelos**: 15 (48% de la unión)
- **Solo DOMINANT**: 8 (26%)
- **Solo GAD-NR**: 8 (26%)
- **Jaccard P95**: 0.484

La señal estructural en G es cualitativamente distinta a la de M. El scatter de percentiles muestra los puntos siguiendo casi exactamente la diagonal y=x en todo el rango — en M había dispersión considerable especialmente en los percentiles medios. La distribución simétrica 26%/48%/26% indica que ningún modelo es sistemáticamente más conservador que el otro en los márgenes.

### 4.2 Comparativa de atributos

- **Correlación Spearman**: ρ = 0.061
- **Solapamiento top-50**: 4/50 (8%)
- **Outliers P95 ambos modelos**: 0 (0% de la unión)
- **Solo DOMINANT**: 23 (50%)
- **Solo GAD-NR**: 23 (50%)
- **Jaccard P95**: 0.000

El Jaccard P95 = 0.000 es el resultado más extremo del experimento: de los 46 grupos que al menos un modelo considera outlier de atributo, ninguno es detectado por ambos. Los 23 de DOMINANT son grupos reales con features atípicas; los 23 de GAD-NR son grupos cuyo `gadnr_feat` es numéricamente el más alto dentro de una distribución degenerada — clasificaciones artefactuales sin valor real.

### 4.3 Scores compuestos

| Estadístico | DOMINANT | GAD-NR |
|---|---|---|
| Media | 2.550 | 0.051 |
| Std | 0.880 | 0.099 |
| Min | 0.800 | 0.004 |
| Max | 5.334 | 0.820 |
| P95 | 4.050 | 0.253 |

- **Correlación Spearman entre compuestos**: ρ = 0.937
- **Solapamiento top-50**: 31/50 (62%)

La correlación de ρ = 0.937 entre scores compuestos es muy superior a M (ρ = 0.575). La razón es directa: `gadnr_score` está dominado por `gadnr_deg` (la única señal con varianza real), y `gadnr_deg` correlaciona casi perfectamente con `dom_struct` (ρ = 0.986). En la práctica, ambos scores compuestos miden esencialmente lo mismo en G: la anomalía estructural del grupo.

### 4.4 Tipologías

| DOMINANT \ GAD-NR | Normal | Estructural | Atributo | Mixta |
|---|---|---|---|---|
| **Normal** | 381 | 6 | 21 | 2 |
| **Estructural** | 8 | 15 | 0 | 0 |
| **Atributo** | 23 | 0 | 0 | 0 |
| **Mixta** | 0 | 0 | 0 | 0 |

Distribución de tipologías:
- DOMINANT: 410 Normal (89.9%), 23 Estructural (5.0%), 23 Atributo (5.0%), 0 Mixta
- GAD-NR: 412 Normal (90.4%), 21 Estructural (4.6%), 21 Atributo (4.6%), 2 Mixta (0.4%)

El acuerdo real entre modelos se concentra exclusivamente en la dimensión estructural: 15 grupos Estructural-Estructural, 381 Normal-Normal. El Jaccard de tipología estructural es 15/(15+8+6) = **0.517**.

**DOMINANT no produce ningún grupo Mixto** — coherente con la correlación negativa entre sus dos señales (ρ = −0.150). Los 2 grupos Mixtos de GAD-NR son artefactos del curriculum: son los grupos con el valor más alto de `gadnr_feat` dentro de una distribución constante.

---

## 5. Señal joint-type de GAD-NR

### 5.1 Caracterización

La señal `gadnr_h` en G tiene rango **0.3521–0.3539** — variación total de 0.0018. Es prácticamente plana. El histograma muestra una distribución unimodal centrada en ~0.353, sin el perfil bimodal que tenía en M.

El P95 cae en 0.353 — prácticamente en el centro de la distribución, no en la cola. Esto significa que el umbral de anomalía no distingue grupos genuinamente anómalos de grupos normales — simplemente corta el 5% superior de una distribución degenerada.

### 5.2 Grupos joint-type puros

Los **23 grupos joint-type puros** (grado medio 10.4 vs 14.4 global) son grupos de conectividad baja-moderada:

Top-10 por `gadnr_h`:
- Thinking Christians Book Club (Religion & Beliefs, grado 11)
- GKIC Nashville Chapter Business Building Meetup (Career & Business, grado 5)
- THE GARDEN OF GOOD (Community & Environment, grado 5)
- The Middle TN Red Sox Nation Meetup Group (Sports & Recreation, grado 5)
- Nashville Professional Photography Meetup by ASMP Tennessee (Career & Business, grado 7)
- Franklin Small Business Digital Media Marketing Meetup (Career & Business, grado 11)
- Nashville Professional Referral Club TEAM Chapter (Career & Business, grado 17)
- 1-on-1 Conversations Nashville (Language & Ethnic Identity, grado 9)
- Cumberland Green Bioregional Council (Community & Environment, grado 1)
- Nashville PowerShell User Group NashPUG (Tech, grado 16)

Predominan grupos de Career & Business, Community & Environment y Religion & Beliefs con grados bajos. No hay ningún patrón temático claro que sugiera anomalía joint-type genuina — los grupos detectados son simplemente los que tienen el valor más alto dentro de una distribución casi constante.

### 5.3 Conclusión sobre la señal joint-type en G

La señal joint-type de GAD-NR no es operativa en el grafo G. Su análisis no aporta información genuina sobre grupos cuyas features son incoherentes con su vecindario. La detección de los 23 grupos "joint-type puros" es un artefacto del curriculum que no debe interpretarse como anomalía real.

---

## 6. Validación contra candidatos del EDA

| Candidato | Tipo EDA | Grado | DOM-struct | DOM-attr | GADNR-deg | GADNR-feat | GADNR-h |
|---|---|---|---|---|---|---|---|
| **Stepping Out Social Dance** | Grado máximo | 91 | #1 (P100) ⚠ | #47 (P90) | #1 (P100) ⚠ | #110 (P76) | #345 (P25) |
| **Nashville Hiking Meetup** | Grupo más grande | 68 | #6 (P99) ⚠ | #37 (P92) | #5 (P99) ⚠ | #55 (P88) | #431 (P6) |
| **NashJS** | Hub tecnológico | 53 | #53 (P89) | #403 (P12) | #16 (P97) ⚠ | #355 (P22) | #284 (P38) |
| **Nashville Children in Nature** | Clustering 0 | 0 | #453 (P1) | #153 (P67) | #429 (P3) | #191 (P58) | #183 (P60) |
| **Sunday Assembly Nashville** | Top betweenness | 62 | #7 (P99) ⚠ | #73 (P84) | #8 (P98) ⚠ | #171 (P63) | #337 (P26) |
| **Lesbians in the Workplace** | Ratio inter-comunidad | 62 | #89 (P81) | #299 (P35) | #129 (P72) | #276 (P40) | #227 (P50) |
| **Christianity & Transhumanism** | Microcomunidad C3 | — | #399 (P13) | #184 (P60) | #346 (P22) | #244 (P47) | #300 (P34) |

### 6.1 Aciertos sólidos

**Stepping Out Social Dance Meetup** es el resultado más rotundo: #1 en ambas señales estructurales (P100 en ambos modelos). El grupo con mayor grado de la red (91) es detectado unánimemente como la anomalía estructural más extrema. Es el caso de validación más limpio de los dos notebooks.

**Nashville Hiking Meetup** (#6 DOM, #5 GADNR, ambos P99+) confirma que el grupo más grande de la red (15.838 miembros) también es estructuralmente anómalo. Su anomalía no viene solo del grado (68, alto pero no el máximo) sino de la combinación de tamaño extremo con una conectividad que ambos modelos difícilmente reconstruyen.

**Sunday Assembly Nashville** (#7 DOM, #8 GADNR, ambos P98+) es el hallazgo más interesante para el TFM: un grupo de Religion & Beliefs en el top-10 estructural de ambos modelos. Confirma el hallazgo del EDA de que este grupo actúa como puente inter-comunidad inusual — su alta betweenness se traduce directamente en anomalía estructural detectable.

### 6.2 Discrepancia informativa — NashJS

NashJS (#53 DOM P89, #16 GADNR P97) es el caso más revelador del notebook. El hub tecnológico extremo — que aparece en 6 de las 10 aristas más pesadas del grafo G — no supera el umbral P95 en DOMINANT pero sí en GAD-NR. Las dos señales estructurales, a pesar de tener ρ = 0.986 globalmente, divergen en este caso concreto. DOMINANT puede reconstruir bien las conexiones específicas de NashJS porque su embedding captura su posición en el grafo; GAD-NR detecta que su grado es inusualmente alto para su posición en el espacio latente. El desacuerdo revela que incluso con ρ casi perfecta las dos señales capturan matices distintos.

### 6.3 Fallos sistemáticos

**Nashville Children in Nature** (grado 0) es invisible para ambas señales estructurales (P1 DOM, P3 GADNR). El mismo patrón que GEEK by AKEIN Engineering en M — un nodo sin vecindario en el grafo no puede ser detectado por ningún modelo GNN+AE. Confirma la limitación arquitectónica como sistemática y no específica de un grafo.

**Lesbians in the Workplace** (ratio inter-comunidad 0.842, el más alto del grafo G) y **Christianity & Transhumanism** (microcomunidad C3 de 4 nodos) no son detectados por ninguna señal. Como se discutió en el análisis del grafo M, el ratio inter-comunidad no implica anomalía de conectividad bruta — un grupo puente tiene conexiones coherentes con su rol, simplemente conecta comunidades distintas. Y la pertenencia a microcomunidad no produce scores estructurales altos cuando el grupo tiene pocas aristas pero internamente coherentes.

---

## 7. Análisis de los embeddings latentes (t-SNE)

### 7.1 Embeddings de DOMINANT

El t-SNE de DOMINANT en G es el resultado visual más claro de los dos notebooks. Tres observaciones:

**Vista por tipo:** los 23 grupos estructurales (azul) forman un cluster compacto y perfectamente separado en la esquina derecha del espacio latente. Los 23 de atributo (naranja) se distribuyen por la izquierda sin ninguna superposición con los estructurales. La ausencia de nodos Mixtos (0 en DOMINANT) es visible — no hay puntos rojos. La separación es mucho más limpia que en el grafo M, donde los clusters eran menos definidos.

**Vista por score:** el gradiente de scores altos (rojo intenso) coincide exactamente con el cluster estructural de la derecha. Los scores decaen suavemente hacia la izquierda. La estructura espacial del score es perfectamente coherente con la tipología.

**Vista por grado:** DOMINANT ha aprendido una representación donde los grupos se ordenan a lo largo de un gradiente continuo de grado — la banda superior del t-SNE corresponde a grupos de grado alto, la inferior a grupos de grado bajo o cero. Este gradiente continuo indica que el embedding captura información estructural rica, no es simplemente una función del grado.

### 7.2 Embeddings de GAD-NR

El t-SNE de GAD-NR en G es cualitativamente mejor que en M. En M el espacio latente era completamente disperso sin estructura; en G aparece un patrón parcialmente interpretable.

**Vista por tipo:** un cluster compacto en la esquina izquierda agrupa los grupos con `gadnr_deg` más alto (anómalos estructurales y los etiquetados como atributo por GAD-NR). Los grupos normales se distribuyen más ampliamente hacia la derecha. La separación no es tan limpia como en DOMINANT pero existe — mejora notable respecto a M.

**Vista por score:** los puntos más rojos se concentran en el cluster izquierdo, coherente con que los scores más altos corresponden a los grupos con mayor `gadnr_deg`.

**Vista por grado:** a diferencia de DOMINANT, GAD-NR no muestra el gradiente continuo de grado. Los grupos de alto grado están dispersos en varias zonas del espacio. El embedding de GAD-NR en G está organizado por `gadnr_deg` (error de reconstrucción de grado), no por el grado real — son cosas distintas.

### 7.3 Comparativa entre modelos

DOMINANT produce embeddings más interpretables en G, al igual que en M. La mayor coherencia del espacio latente de DOMINANT en G (separación perfecta de tipologías, gradiente continuo de grado) frente al de M (clusters menos definidos) refleja que la mayor densidad del grafo G facilita el aprendizaje de representaciones estructurales más ricas.

---

## 8. Comparativa M vs G

Esta es la sección más relevante para el capítulo de discusión del TFM, ya que permite contextualizar los resultados de G en relación con M.

| Métrica | Grafo M | Grafo G |
|---|---|---|
| ρ señal estructural (DOM vs GADNR) | 0.668 | **0.986** |
| Jaccard P95 estructural | ~0.49 | **0.484** |
| ρ señal atributo (DOM vs GADNR) | 0.447 | **0.061** |
| Jaccard P95 atributo | 0.02 | **0.000** |
| `gadnr_feat` operativa | Sí (0.04–0.24) | No (0.000) |
| `gadnr_h` operativa | Sí (0.94–3.55) | No (0.352–0.354) |
| Lambdas finales curriculum | 0.01/7.6/2.4·10⁻⁵ | 0.01/7.6/2.4·10⁻⁵ |
| Inestabilidad GAD-NR | No | Sí → lr=0.001 |
| ρ scores compuestos | 0.575 | 0.937 |
| t-SNE DOMINANT interpretable | Sí | Muy sí |
| t-SNE GAD-NR interpretable | No | Parcialmente |

**Señal estructural:** mejora drásticamente de M a G (ρ 0.668→0.986). La mayor densidad del grafo G (0.065 vs 0.018) hace que la reconstrucción de adyacencia sea más determinista y los dos modelos converjan a rankings casi idénticos. El Jaccard P95 se mantiene similar (~0.48-0.49) porque el número de outliers detectados por cada modelo es proporcional al grafo.

**Señal de atributo:** se degrada completamente de M a G (ρ 0.447→0.061, Jaccard 0.02→0.000). Con 35 features el curriculum sobreoptimiza hasta anular la señal. La paradoja es que más features produce peor señal de atributo en GAD-NR.

**Curriculum invariante:** los lambdas finales son idénticos en M y G a pesar de sus diferencias. El mecanismo de PyGOD converge siempre al mismo punto fijo. Esto es una limitación de la librería documentada empíricamente en el TFM.

**Inestabilidad en G:** la mayor dimensionalidad de features produce inestabilidad numérica con `lr=0.004` que no aparecía en M. El ajuste a `lr=0.001` resuelve el problema pero introduce una asimetría en la configuración entre grafos que debe documentarse.

**Scores compuestos más correlacionados en G** (ρ 0.575→0.937) porque en G ambos scores están esencialmente midiendo lo mismo — la señal estructural — al no haber señal de atributo operativa en GAD-NR.

---

## 9. Limitaciones identificadas

### 9.1 Limitaciones del grafo G

1. **146 grupos excluidos.** Los grupos sin co-membresía con ningún otro quedan fuera del grafo G y son invisibles para el análisis. Son precisamente los grupos más aislados — posibles anomalías estructurales extremas — que el modelo no puede evaluar.

2. **Señal de atributo limitada a DOMINANT.** La única señal de atributo informativa en G es `dom_attr`. La riqueza semántica de las 35 features (categoría temática, tamaño, actividad) queda parcialmente desaprovechada porque GAD-NR no puede explotarla.

3. **Ratio inter-comunidad no detectable.** Los grupos puente entre comunidades temáticamente distintas (Lesbians in the Workplace, Nashvegans, Sunday Assembly en sus roles de puente) no son detectados como anómalos por ninguna señal, porque su conectividad es coherente con su rol aunque sea inusual. Para detectar este tipo de anomalía sería necesario incorporar la pertenencia a comunidad como feature del nodo, lo cual entraría en contradicción con el diseño metodológico del TFM.

### 9.2 Limitaciones de los modelos

1. **GAD-NR inoperante en atributo y joint-type.** Las dos señales distintivas de GAD-NR respecto a DOMINANT (`gadnr_feat` y `gadnr_h`) son completamente inútiles en G. En la práctica, GAD-NR aporta solo `gadnr_deg` — una señal equivalente a `dom_struct` con ρ = 0.986.

2. **Nodos sin vecindario invisibles.** Nashville Children in Nature (grado 0) no puede ser detectado por ningún modelo. Limitación arquitectónica sistemática.

3. **Inestabilidad numérica de GAD-NR.** El training con `lr=0.004` produce NaN en grafos con alta dimensionalidad de features. Requiere ajuste específico por grafo que introduce asimetría metodológica.

4. **Curriculum de PyGOD no adaptativo.** El mecanismo de re-ponderación de lambdas converge siempre al mismo punto fijo, independientemente de las características del grafo. Los lambdas declarados por el usuario no tienen efecto real en los lambdas efectivos al final del entrenamiento.

### 9.3 Limitaciones metodológicas

1. **Comparativa limitada en dimensión de atributo.** Con `gadnr_feat` constante, la comparativa DOMINANT vs GAD-NR en G se reduce esencialmente a una comparativa de señales estructurales. La riqueza metodológica del diseño (cinco señales independientes) se reduce a tres en la práctica.

2. **K=50 en solapamientos.** Con 456 grupos, el top-50 representa el 11% del grafo — un umbral menos restrictivo que el 0.88% del top-100 en M. Las comparativas directas de solapamiento entre grafos deben tener esto en cuenta.

---

## 10. Conclusiones del notebook

Los resultados del notebook G confirman y complementan los del notebook M, con hallazgos nuevos en ambas direcciones:

**Lo que mejora respecto a M:**
La señal estructural es mucho más consistente entre modelos (ρ = 0.986 vs 0.668). Los candidatos del EDA más prominentes (Stepping Out, Nashville Hiking, Sunday Assembly) son detectados por ambos modelos simultáneamente en el top-10, con una claridad que en M no se conseguía. Los embeddings de DOMINANT son los más interpretables de los dos notebooks. El desacuerdo informativo de NashJS (P89 DOM vs P97 GADNR) muestra que incluso con correlación casi perfecta las dos señales capturan matices distintos.

**Lo que empeora respecto a M:**
La señal de atributo de GAD-NR queda completamente anulada. La señal joint-type, que en M era la aportación más interesante de GAD-NR, es inoperante en G. GAD-NR en G es esencialmente un detector de anomalías estructurales con un decoder de grado — su ventaja arquitectónica sobre DOMINANT desaparece.

**El hallazgo más relevante para el TFM:**
El curriculum de PyGOD converge siempre a los mismos lambdas finales (0.01/7.6/2.4·10⁻⁵) independientemente del grafo. Este comportamiento, documentado empíricamente en M y confirmado en G, es una limitación de la librería que condiciona profundamente la interpretación de los resultados de GAD-NR y que cualquier usuario de PyGOD debería conocer.

### 10.1 Hipótesis para el grafo MG (bipartito)

El grafo MG tiene 25.233 nodos, 45.583 aristas y 38 features por nodo (3 de miembro + 35 de grupo con zero-padding). Basándose en los resultados de M y G, las hipótesis son:

- El curriculum producirá los mismos lambdas finales (0.01/7.6/2.4·10⁻⁵).
- La señal de atributo de GAD-NR será igualmente inoperante — 38 features con zero-padding son aún más ricas que las 35 de G.
- La señal estructural puede ser menos coherente entre modelos que en G (grafo más grande y heterogéneo) pero más coherente que en M (estructura más rica que el grafo de miembros).
- La heterogeneidad de nodos (dos tipos con perfiles de features completamente distintos) puede producir clusters diferenciados en el t-SNE que no aparecen en M ni en G.
- Nashville Children in Nature y grupos aislados seguirán siendo invisibles para los modelos.

---

## 11. Resumen ejecutivo

**Grafo:** G — 456 grupos, 6.692 aristas únicas, 35 features (categoría, tamaño, actividad).

**DOMINANT:** convergencia perfecta, ρ verificación = 1.000000. Sub-scores `dom_struct` (0.87–8.88) y `dom_attr` (0.14–2.81) ortogonales (ρ = −0.150). Señal de atributo informativa y única en G.

**GAD-NR:** requirió lr=0.001 por inestabilidad NaN con lr=0.004. Lambdas finales idénticos a M (0.01/7.6/2.4·10⁻⁵). `gadnr_feat` completamente degenerada (rango 0.000). `gadnr_h` prácticamente plana (rango 0.0018). Solo `gadnr_deg` (3.47–33.439) es informativa.

**Señal estructural:** ρ = 0.986, Jaccard P95 = 0.484. La señal más consistente de los dos notebooks. Stepping Out (#1/#1), Nashville Hiking (#6/#5), Sunday Assembly (#7/#8) validados por ambos modelos.

**Señal de atributo:** ρ = 0.061, Jaccard P95 = 0.000. Completamente divergente. Solo DOMINANT aporta señal real de atributo en G.

**Joint-type:** inoperante en G. Los 23 grupos identificados son artefactos del curriculum.

**Validación EDA:** 3 de 7 candidatos validados por ambos modelos (Stepping Out, Nashville Hiking, Sunday Assembly). NashJS como desacuerdo informativo. Nashville Children in Nature, Lesbians in the Workplace y Christianity & Transhumanism sin detección.

**Hallazgo clave:** el curriculum de PyGOD es invariante al grafo — mismos lambdas finales en M y G a pesar de sus diferencias. Con más features, las consecuencias son más extremas y anulan las señales de atributo y joint-type completamente.

**Salidas:** `scores_G.csv` (456 grupos × 21 columnas) y `embeddings_G.npz`.
