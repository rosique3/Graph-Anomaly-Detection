# Objetivos del TFM

## Título

**Detección de anomalías no supervisada en redes sociales atribuidas mediante Graph Autoencoders: un estudio comparativo sobre el dataset Meetup Tennessee**

---

## Contexto

Este TFM se enmarca en el campo de la detección de anomalías en grafos atribuidos — redes donde los nodos tienen tanto una estructura de conexiones como un conjunto de atributos o features asociados. La detección de anomalías en este tipo de redes es un problema relevante en múltiples dominios (fraude, ciberseguridad, redes sociales) y metodológicamente complejo porque las anomalías pueden manifestarse en tres dimensiones distintas y no necesariamente correlacionadas: la estructura de conexiones del nodo, sus atributos individuales, o la incoherencia entre ambos.

El dataset de Meetup Tennessee (Kaggle, 2015–2017) proporciona datos reales de una red social de eventos presenciales en Nashville, Tennessee. Su riqueza radica en que permite construir tres representaciones complementarias en grafo — miembros, grupos y bipartito miembro-grupo — cada una con semántica distinta y candidatos a anomalía identificables a priori mediante análisis exploratorio.

---

## Objetivo general

Desarrollar y evaluar un pipeline de detección de anomalías no supervisada sobre redes sociales atribuidas, comparando dos arquitecturas de Graph Autoencoder — DOMINANT y GAD-NR — sobre tres representaciones en grafo del dataset Meetup Tennessee, y analizando cualitativamente los resultados contra candidatos a anomalía identificados en el análisis exploratorio previo.

---

## Objetivos específicos

### 1. Análisis exploratorio del dataset

Caracterizar exhaustivamente el dataset de Meetup Tennessee en sus dimensiones tabular y estructural, identificando candidatos a anomalía en cada representación antes de aplicar ningún modelo. Este análisis previo sirve como ground truth cualitativo para la validación posterior de los modelos.

- Análisis de metadatos: distribuciones geográficas, calidad de datos, patrones de organizadores y eventos.
- Análisis de grafos: métricas estructurales (grado, clustering, betweenness, comunidades), distribución de pesos, perfiles de comportamiento en el grafo bipartito.
- Identificación de candidatos a anomalía por tipo: estructural (super-conectores, nodos puente, nodos aislados), atributo (grupos fantasma, miembros con features atípicas) y mixta.
- Visualización de los tres grafos mediante Gephi con el algoritmo ForceAtlas 2, tanto en su representación base (tamaño de nodo proporcional al grado) como coloreados por comunidades Louvain, como complemento visual al análisis estructural. Se justifica la elección de ForceAtlas 2 frente a alternativas como Fruchterman-Reingold u OpenOrd en términos de escalabilidad para grafos grandes y densos, respeto a los pesos de arista y emergencia natural de comunidades — la coincidencia entre las regiones espaciales del layout y las comunidades Louvain se emplea como evidencia visual de que las comunidades detectadas reflejan estructuras reales del grafo.

### 2. Diseño del pipeline de feature engineering

Construir las matrices de features `X` para cada grafo a partir de los metadatos disponibles, siguiendo criterios metodológicos explícitos:

- Usar únicamente atributos de metadatos originales, sin métricas estructurales derivadas del grafo (para evitar redundancia con la información que los modelos aprenden de `A`).
- Diseñar features interpretables y justificadas para cada tipo de nodo.
- Resolver la heterogeneidad del grafo bipartito mediante zero-padding.

### 3. Implementación y entrenamiento de los modelos

Implementar dos detectores de anomalías de la librería PyGOD 1.1.0 sobre cada uno de los tres grafos:

- **DOMINANT** (Ding et al., SDM 2019): baseline clásico con encoder GCN compartido y dos decoders independientes para estructura y atributos.
- **GAD-NR** (Roy et al., WSDM 2024): estado del arte con reconstrucción de vecindario completo mediante tres señales independientes (features propias, grado y distribución KL del vecindario).

Documentar el proceso de entrenamiento incluyendo decisiones de hiperparámetros, problemas encontrados y workarounds aplicados.

### 4. Extracción y análisis de señales de anomalía

Extraer y analizar los sub-scores individuales de cada modelo como señales independientes, en lugar de trabajar exclusivamente con el score compuesto agregado:

- `dom_struct` y `dom_attr` para DOMINANT.
- `gadnr_deg`, `gadnr_feat` y `gadnr_h` para GAD-NR.

Esta decisión metodológica — justificada empíricamente por el comportamiento del curriculum interno de PyGOD — permite preservar la distinción semántica entre tipos de anomalía y evaluar cada dimensión de forma independiente.

### 5. Comparativa entre modelos

Analizar las similitudes y diferencias entre DOMINANT y GAD-NR en cada grafo, sin el objetivo de declarar un modelo ganador sino de caracterizar la información complementaria que aporta cada uno:

- Correlaciones Spearman entre señales equivalentes (estructural vs estructural, atributo vs atributo).
- Solapamiento de rankings extremos (top-K y Jaccard P95).
- Acuerdo en tipología asignada (crosstab de tipos de anomalía).
- Comparativa de embeddings latentes mediante t-SNE.

### 6. Validación cualitativa contra candidatos del EDA

Verificar si los modelos recuperan los candidatos a anomalía identificados en el análisis exploratorio, documentando aciertos, discrepancias informativas y fallos sistemáticos:

- **Aciertos sólidos**: candidatos detectados por ambos modelos con alta consistencia.
- **Discrepancias informativas**: candidatos donde los modelos discrepan, revelando que cada señal captura una dimensión distinta de la anomalía.
- **Fallos sistemáticos**: candidatos que ningún modelo detecta, con justificación de la limitación arquitectónica subyacente (nodos sin vecindario, anomalías semánticas no capturables por GNN+AE).

### 7. Análisis enriquecido del grafo bipartito

Aprovechar las características únicas del grafo bipartito miembro-grupo para análisis adicionales no posibles en M ni G:

- Distribución de scores por tipo de nodo (miembros vs grupos).
- Cruce de scores con perfiles de comportamiento del EDA (inactivos, exploradores, fieles, hiperactivos).
- Cruce de scores con ratio de actividad de grupos (grupos fantasma, grupos con baja participación real).
- Análisis de la relación entre actividad real (peso de aristas = eventos asistidos) y score de anomalía.

### 8. Documentación de limitaciones y hallazgos metodológicos

Documentar con rigor las limitaciones observadas durante la experimentación, distinguiendo entre limitaciones del dataset, de los modelos y de la implementación:

- **Limitación del dataset**: pocas features geográficas en el grafo M, heterogeneidad de nodos en MG con zero-padding.
- **Limitación arquitectónica**: DOMINANT es O(n²) en memoria por la matriz densa, GAD-NR no está diseñado para grafos bipartitos heterogéneos.
- **Limitación de implementación**: el curriculum interno de PyGOD 1.1.0 reescala los lambdas de GAD-NR independientemente de los valores declarados, produciendo siempre los mismos lambdas efectivos (0.01/7.6/2.4·10⁻⁵) con independencia del grafo y de los lambdas iniciales.

---

## Enfoque metodológico

El TFM adopta un enfoque de **detección de anomalías no supervisada** — no existe ground truth etiquetado y la evaluación es exclusivamente cualitativa. Esto es coherente con la naturaleza del problema real: en redes sociales reales no se dispone de etiquetas de "anomalía confirmada", y el valor del sistema está en su capacidad de señalar candidatos que merecen inspección, no en optimizar una métrica supervisada.

La validación se basa en la **coherencia entre los resultados de los modelos y las expectativas fundadas en el análisis exploratorio**. Un candidato identificado como anómalo por el EDA y confirmado por ambos modelos es una validación sólida. Una discrepancia entre modelos es informativa sobre las dimensiones distintas que cada arquitectura captura. Un fallo sistemático documentado y justificado es una contribución metodológica honesta.

---

## Dataset y representaciones en grafo

| Grafo | Nodos | Aristas | Features/nodo | Semántica de arista |
|---|---|---|---|---|
| **M** — Miembros | 11.371 | 1.176.024 | 3 (geográficas) | Grupos compartidos |
| **G** — Grupos | 456 | 6.692 | 35 (categoría, tamaño, actividad) | Miembros compartidos |
| **MG** — Bipartito | 25.233 | 45.583 | 38 (heterogéneo con zero-padding) | Eventos asistidos |

---

## Modelos implementados

| Modelo | Año | Señales extraídas | Tipo de anomalía |
|---|---|---|---|
| **DOMINANT** | SDM 2019 | `dom_struct`, `dom_attr` | Estructural, Atributo |
| **GAD-NR** | WSDM 2024 | `gadnr_deg`, `gadnr_feat`, `gadnr_h` | Estructural, Atributo, Joint-type |

---

## Estructura de la memoria

La memoria sigue el índice acordado con la universidad:

1. **Introducción**
   - 1.1 Introducción
   - 1.2 Motivación
   - 1.3 Objetivos
   - 1.4 Organización de la memoria

2. **Estado del arte**
   - 2.1 Análisis de aplicaciones similares
   - 2.2 Tecnologías

3. **Requisitos, especificaciones, coste, riesgos, viabilidad**
   - 3.1 Requisitos
   - 3.2 Especificaciones
   - 3.3 Costes
   - 3.4 Riesgos
   - 3.5 Viabilidad

4. **Análisis**
   - 4.1 Dataset: Meetup Tennessee
     - 4.1.1 Descripción y estructura
     - 4.1.2 Representación en grafo
   - 4.2 EDA tabular
     - 4.2.1 meta_members
     - 4.2.2 meta_groups
     - 4.2.3 meta_events
   - 4.3 EDA de grafos
     - 4.3.1 Grafo de miembros (M)
     - 4.3.2 Grafo de grupos (G)
     - 4.3.3 Grafo bipartito (MG)
     - 4.3.4 Visualización con Gephi
   - 4.4 Candidatos a anomalía identificados en el EDA

5. **Diseño**
   - 5.1 Feature engineering
     - 5.1.1 Features de nodos miembro
     - 5.1.2 Features de nodos grupo
     - 5.1.3 Matriz unificada para el bipartito
     - 5.1.4 Normalización
   - 5.2 Arquitectura de los modelos
     - 5.2.1 DOMINANT — modelo clásico de referencia
     - 5.2.2 GAD-NR — modelo reciente
     - 5.2.3 Comparación teórica entre ambos
   - 5.3 Estrategia de experimentación
     - 5.3.1 Un experimento por grafo y modelo (6 en total)
     - 5.3.2 Hiperparámetros
     - 5.3.3 Métricas de evaluación cualitativa

6. **Implementación y pruebas**
   - 6.1 Implementación
   - 6.2 Pruebas funcionales
     - 6.2.1 Grafo de miembros (M)
     - 6.2.2 Grafo de grupos (G)
     - 6.2.3 Grafo bipartito (MG)
   - 6.3 Pruebas de rendimiento
   - 6.4 Validación
   - 6.5 Discusión

7. **Conclusiones**
   - 7.1 Revisión de costes
   - 7.2 Conclusiones
   - 7.3 Trabajo futuro

**Apéndice**
- A.1 Código fuente
- A.2 Tablas complementarias
- A.3 Figuras adicionales

**Bibliografía**

Los capítulos 4, 5 y 6 concentran el trabajo técnico. El capítulo 6 es el más extenso e incluye los resultados de los seis experimentos (dos modelos × tres grafos) con sus análisis comparativos y validaciones cualitativas. El capítulo 7 recoge las conclusiones generales, la revisión del coste real frente al estimado y las líneas de trabajo futuro.