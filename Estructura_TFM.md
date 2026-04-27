# Estructura del TFM — Detección de Anomalías en Grafos

## 1. Introducción
- 1.1 Motivación y contexto
- 1.2 Objetivos del trabajo
- 1.3 Estructura del documento

## 2. Estado del Arte
- 2.1 Detección de anomalías: visión general
- 2.2 Detección de anomalías en grafos
- 2.3 Autoencoders para detección de anomalías
- 2.4 Graph Autoencoders (GAE y VGAE)
- 2.5 Trabajos relacionados

## 3. Dataset
- 3.1 Descripción del dataset (Meetup Tennessee)
- 3.2 Estructura de los datos
- 3.3 Representación en grafo
  - 3.3.1 Introducción al EDA de grafos
  - 3.3.2 Justificación del análisis separado por grafo
  - 3.3.3 Inconsistencias entre grafos

## 4. Análisis Exploratorio de Datos (EDA)
- 4.1 EDA tabular
  - 4.1.1 meta_members
  - 4.1.2 meta_groups
  - 4.1.3 meta_events
- 4.2 EDA de grafos
  - 4.2.1 Grafo de miembros
  - 4.2.2 Grafo de grupos
  - 4.2.3 Grafo bipartito miembro-grupo

## 5. Feature Engineering
- 5.1 Features estructurales por grafo
  - 5.1.1 Features del grafo de miembros
  - 5.1.2 Features del grafo de grupos
  - 5.1.3 Features del grafo bipartito
- 5.2 Consolidación del conjunto de features
- 5.3 Normalización y preprocesado

## 6. Metodología
- 6.1 Formulación del problema
- 6.2 Graph Autoencoder (GAE)
  - 6.2.1 Fundamentos teóricos
  - 6.2.2 Arquitectura del modelo
  - 6.2.3 Función de pérdida y anomaly score
- 6.3 Variational Graph Autoencoder (VGAE)
  - 6.3.1 Fundamentos teóricos
  - 6.3.2 Diferencias con GAE
- 6.4 Estrategia de experimentación
  - 6.4.1 Configuración de experimentos
  - 6.4.2 Hiperparámetros
  - 6.4.3 Métricas de evaluación

## 7. Experimentos y Resultados
- 7.1 Detección de anomalías en el grafo de miembros
  - 7.1.1 Configuración del experimento
  - 7.1.2 Resultados
  - 7.1.3 Análisis de anomalías detectadas
- 7.2 Detección de anomalías en el grafo de grupos
  - 7.2.1 Configuración del experimento
  - 7.2.2 Resultados
  - 7.2.3 Análisis de anomalías detectadas
- 7.3 Detección de anomalías en el grafo bipartito
  - 7.3.1 Configuración del experimento
  - 7.3.2 Resultados
  - 7.3.3 Análisis de anomalías detectadas
- 7.4 Comparativa entre grafos

## 8. Discusión
- 8.1 Interpretación de resultados
- 8.2 Anomalías detectadas vs. candidatos identificados en el EDA
- 8.3 Limitaciones del estudio
- 8.4 Implicaciones prácticas

## 9. Conclusiones
- 9.1 Conclusiones principales
- 9.2 Contribuciones del trabajo
- 9.3 Trabajo futuro

## Referencias

## Anexos
- Anexo A: Código fuente
- Anexo B: Tablas complementarias
- Anexo C: Figuras adicionales

---

Algunas observaciones sobre esta estructura:

**El capítulo 5 (Feature Engineering)** puede sorprenderte — lo incluyo porque aunque el GAE aprende de la estructura del grafo, necesitas definir las features de nodo X. Esas features pueden ser las métricas estructurales que ya calculamos (grado, betweenness, clustering, comunidad...) enriquecidas con metadatos.

**El capítulo 7** está pensado para comparar los tres grafos — esto le da mucho valor al TFM porque demuestras que cada grafo detecta tipos de anomalías distintos.

**La sección 8.2** es clave para un TFM riguroso — contrastar lo que el modelo detecta con los candidatos que ya identificamos en el EDA da mucha solidez a los resultados.

¿Te parece bien esta estructura o quieres modificar algo?