# EDA de Grafos — Meetup Tennessee

## 1. Introducción

### 1.1 Descripción general

El dataset de Meetup Tennessee permite construir tres representaciones en grafo de la misma red social, cada una capturando una dimensión distinta de las relaciones entre sus elementos. Estas tres representaciones no son redundantes sino complementarias — cada una responde a preguntas analíticas distintas y aporta información estructural que las otras no contienen.

Los tres grafos son los siguientes:

- **Grafo de miembros** (`member-edges.csv`): grafo no dirigido donde los nodos son miembros y las aristas conectan pares de miembros que pertenecen a al menos un grupo común.
- **Grafo de grupos** (`group-edges.csv`): grafo no dirigido donde los nodos son grupos y las aristas conectan pares de grupos que comparten al menos un miembro.
- **Grafo bipartito miembro-grupo** (`member-to-group-edges.csv`): grafo bipartito no dirigido donde los nodos son miembros y grupos, y las aristas representan la pertenencia de un miembro a un grupo.

### 1.2 Significado del peso de las aristas

Un aspecto fundamental que justifica el análisis separado de los tres grafos es que el **peso de las aristas tiene un significado completamente distinto** en cada uno de ellos:

En el **grafo de miembros**, el peso de una arista entre dos miembros representa el **número de grupos que comparten**. Es una medida de afinidad social — cuantos más grupos tienen en común dos miembros, más fuerte es su vínculo en la red. Un peso alto indica una relación de co-membresía muy sólida y sugiere intereses compartidos en múltiples ámbitos.

En el **grafo de grupos**, el peso de una arista entre dos grupos representa el **número de miembros que comparten**. Es una medida de solapamiento de audiencia — cuantos más miembros tienen en común dos grupos, más próximos están en el espacio social. Un peso alto indica que ambos grupos atraen a un perfil de miembro similar y probablemente pertenecen a la misma comunidad temática.

En el **grafo bipartito**, el peso de una arista entre un miembro y un grupo representa el **número de eventos del grupo a los que ha asistido ese miembro**. Es una medida de actividad real — a diferencia de los otros dos grafos, donde el peso es una métrica estructural derivada de la co-membresía, aquí el peso refleja el comportamiento concreto y observado del miembro dentro del grupo. Un peso alto indica un miembro muy activo en ese grupo, mientras que un peso bajo puede indicar un miembro que se apuntó pero apenas participó.

Esta diferencia es especialmente relevante para la detección de anomalías: el grafo bipartito es el único de los tres que permite distinguir entre un miembro que pertenece a un grupo y un miembro que realmente participa en él, lo que abre la puerta a detectar patrones de comportamiento anómalos como miembros que se apuntan masivamente a grupos sin asistir a ningún evento.

### 1.3 Justificación del análisis separado

La decisión de analizar los tres grafos de forma independiente responde a tres razones principales.

La primera es la **heterogeneidad semántica** ya descrita: dado que el peso de las aristas mide cosas distintas en cada grafo, consolidarlos en un único grafo mezclado produciría aristas sin interpretación clara y métricas sin significado.

La segunda es la **diferencia en la naturaleza de los nodos**: el grafo bipartito es el único que contiene simultáneamente nodos de dos tipos distintos (miembros y grupos), lo que implica que muchas métricas estándar de grafos — como el coeficiente de clustering, que requiere la existencia de triángulos — no son aplicables directamente y requieren adaptaciones específicas para grafos bipartitos.

La tercera es la **complementariedad analítica**: cada grafo ilumina una dimensión distinta de la red. El grafo de miembros es el más adecuado para analizar la estructura social y detectar anomalías a nivel de individuo. El grafo de grupos es el más adecuado para analizar la estructura temática y detectar anomalías a nivel de comunidad. El grafo bipartito es el más completo y el más adecuado para analizar el comportamiento de participación y detectar anomalías basadas en la actividad real.

### 1.4 Inconsistencias entre grafos: limitaciones del dataset

Al comparar el número de nodos entre los tres grafos y los metadatos originales, se observan inconsistencias significativas que es necesario documentar y tener en cuenta en cualquier análisis posterior.

| Fuente | Nodos miembro | Nodos grupo |
|---|---|---|
| `meta_members` | 24.591 | — |
| `meta_groups` | — | 602 |
| Grafo bipartito (`MG`) | 24.631 | 602 |
| Grafo de miembros (`M`) | 11.372 | — |
| Grafo de grupos (`G`) | — | 456 |

**Inconsistencia en los nodos de miembro.** El grafo bipartito contiene 24.631 nodos de miembro, mientras que `meta_members` registra 24.591 — la diferencia de 40 miembros corresponde a miembros presentes en `member-to-group-edges.csv` sin correspondencia en los metadatos, lo que constituye una anomalía de integridad referencial. La inconsistencia más importante es la que existe entre el grafo bipartito (24.631 nodos) y el grafo de miembros (11.372 nodos): solo el 46.2% de los miembros del bipartito aparecen en el grafo de miembros. La causa fundamental es que **`member-edges.csv` únicamente registra pares de miembros que tienen co-membresía — no incluye nodos aislados**. Los aproximadamente 13.000 miembros ausentes pertenecen a grupos pero no comparten esa membresía con ningún otro miembro registrado en el dataset, por lo que no generan ninguna arista.

**Inconsistencia en los nodos de grupo.** El grafo bipartito contiene 602 nodos de grupo, cifra que coincide exactamente con `meta_groups`. El grafo de grupos, en cambio, contiene únicamente 456 nodos — los 146 grupos ausentes (24.3%) no aparecen en `group-edges.csv` por la misma razón estructural: **`group-edges.csv` solo registra pares de grupos que comparten al menos un miembro, sin incluir nodos aislados**.

**No conexidad del grafo bipartito.** A diferencia del grafo de miembros y del grafo de grupos, que son completamente conexos, el grafo bipartito no es conexo. Esto es consistente con todo lo anterior: el bipartito incluye a todos los miembros y grupos, entre ellos aquellos que no comparten co-membresía con ningún otro nodo. La componente principal concentra la gran mayoría de los nodos (99.8%) y es sobre ella donde tiene sentido realizar los análisis estructurales.

**Decisión metodológica.** Ante estas inconsistencias, se opta por mantener los tres grafos con sus nodos originales sin forzar ningún tipo de filtrado o enriquecimiento artificial. Cada grafo responde a una pregunta analítica distinta y los análisis realizados sobre cada uno son internamente consistentes. Forzar la homogeneización implicaría añadir nodos aislados que distorsionarían las métricas globales sin aportar información real. Además, las inconsistencias son en sí mismas un hallazgo relevante: los miembros y grupos ausentes de sus respectivos grafos son los candidatos más claros a anomalía estructural del dataset.

---

## 2. Grafo de Miembros

### 2.1 Estadísticas básicas y métricas globales

El grafo de miembros contiene **11.372 nodos y 1.176.368 aristas**. Solo el **46.2% de los miembros de `meta_members` aparecen en el grafo** — el resto no comparte membresía con ningún otro miembro y por tanto no genera aristas. El grafo es **completamente conexo** (una única componente).

La **densidad es de 0.0182**, lo que lo clasifica como un grafo disperso, comportamiento habitual y esperable en redes sociales reales de este tamaño. Sin embargo, el **grado medio de 206.89** es llamativamente alto — cada miembro comparte membresía con otros 207 de media — consecuencia directa de la existencia de grupos gigantes como Nashville Hiking Meetup que conectan a miles de miembros entre sí de golpe. El dato más llamativo es el **coeficiente de clustering medio de 0.885**, un valor muy alto para una red de este tamaño, que refleja una estructura de comunidad muy fuerte.

### 2.2 Distribución de grado

La distribución de grado presenta una **fuerte asimetría positiva** característica de redes sociales, con una media (207) muy superior a la mediana (147) y una desviación típica (202) prácticamente igual a la media. El rango va de 1 a 2.356. La representación en escala log-log no muestra una ley de potencia limpia, sino una nube dispersa con varias bandas horizontales — consecuencia de la naturaleza discreta del grafo: pertenecer a un grupo de 500 miembros añade automáticamente ~500 conexiones de golpe.

La segmentación de nodos por grado revela la siguiente estructura: 13 nodos hoja (grado = 1), 2.847 nodos de grado bajo (2-65), 5.773 nodos de grado medio (66-283) — el segmento mayoritario con el 50.8% del total —, 2.642 nodos de grado alto (284-1.000) y 97 super-conectores con grado superior a 1.000.

### 2.3 Análisis de nodos extremos

**Jim H** es el miembro más conectado de toda la red con un grado de 2.356, casi 450 conexiones más que el segundo (Shalini, 1.911). Los 20 miembros con mayor grado son casi exclusivamente residentes en Nashville, TN.

En el extremo opuesto, los **13 nodos hoja** presentan perfiles heterogéneos. Destaca **"GEEK by AKEIN Engineering"**, claramente una entidad empresarial y no una persona física, lo que constituye una anomalía de tipo de entidad en la red.

### 2.4 Métricas de centralidad

El análisis de degree centrality y betweenness centrality revela patrones estructurales muy relevantes. **Jim H lidera en degree centrality (0.207) pero no en betweenness (0.011)** — está muy conectado pero sus vecinos también se conocen entre sí, por lo que no actúa como puente entre comunidades. **Shalini** presenta el caso más interesante: segunda en degree centrality pero primera en betweenness (0.021), lo que la convierte en el puente más importante de la red. Más llamativo aún es el caso de **Pablo**, que con una degree centrality moderada (posición 756 en el ranking) alcanza la tercera posición en betweenness (0.018) — un rol estructural de puente desproporcionadamente alto respecto a su número de conexiones.

El dato más significativo es que **8.241 nodos (72.5%) tienen betweenness = 0**, confirmando que la gran mayoría de miembros están completamente embebidos dentro de sus comunidades.

### 2.5 Coeficiente de clustering local

La distribución del clustering local es marcadamente **bimodal**: **8.228 nodos (72.4%) tienen clustering = 1.0** — sus vecinos forman un clique perfecto — mientras que un grupo minoritario presenta valores entre 0.4 y 0.9. Los **13 nodos con clustering = 0** coinciden exactamente con los 13 nodos hoja, lo cual es matemáticamente inevitable: con un único vecino no puede existir ningún triángulo. Los **922 nodos con clustering < 0.5** son los más relevantes para detección de anomalías: tienen suficientes conexiones para que la métrica sea significativa, pero sus vecinos no se conocen entre sí, indicando un rol de puente entre comunidades distintas.

### 2.6 Análisis de pesos de aristas

Las aristas representan el número de grupos que dos miembros tienen en común. La distribución es extremadamente concentrada: **el 95.9% de las aristas tienen peso = 1**, con media (1.05) prácticamente igual a la mediana (1.0) y peso máximo de 9. Las **74 aristas con peso > 5** son candidatas a anomalía por posible co-membresía coordinada. El par más fuerte de toda la red es **Garrett Vangilder — Matt Kraatz** con 9 grupos compartidos.

### 2.7 Detección de comunidades

Se aplicaron **Louvain** y **Greedy Modularity**. Louvain obtuvo una modularidad de **0.6755** frente a los 0.6127 de Greedy, detectando 22 comunidades frente a las 35 de Greedy. Una modularidad superior a 0.6 indica comunidades bien definidas y separadas. Louvain se adopta como algoritmo principal por su mejor rendimiento y menor fragmentación.

| Métrica | Louvain | Greedy |
|---|---|---|
| Nº comunidades | 22 | 35 |
| Modularidad | 0.6755 | 0.6127 |
| Tamaño medio | 516.9 | 324.9 |
| Comunidad más grande | 2.215 | 2.622 |
| Comunidades < 10 nodos | 1 | 11 |

Las 22 comunidades presentan una distribución de tamaños muy heterogénea, desde 6 hasta 2.215 nodos, con temáticas dominantes bien definidas en la mayoría. **C11 (Tech, 2.215 nodos)** es la más grande. **C7 (Outdoors & Adventure, 927 nodos)** alberga a Shalini — el mayor puente de la red — como miembro más conectado. **C17 (Dancing, 772 nodos)** alberga a Jim H. Desde el punto de vista de anomalías, **C6 (Socializing, 508 nodos)** es la comunidad estructuralmente más influyente pese a su tamaño moderado, mientras que **C2 (13 nodos)** y **C21 (6 nodos)** son microcomunidades completamente periféricas con betweenness = 0.

### 2.8 Análisis de nodos puente entre comunidades

El ratio inter-comunidad — fracción de vecinos de un nodo en comunidades distintas a la suya — tiene una **media de 0.181**. Hay **211 nodos con ratio = 0** (completamente embebidos) y **487 nodos con ratio > 0.5** (verdaderos puentes). El scatter de degree vs. ratio revela que **a mayor degree, menor ratio inter-comunidad** — los super-conectores tienden a estar más embebidos en su comunidad. La excepción más notable es **Jim H** (degree 0.207, ratio 0.759), que ejerce simultáneamente un rol de super-conector y puente inter-comunidad. Los casos más interesantes son los nodos con degree baja pero ratio alto: **Tremaine James** (ratio 0.833) y **Mary Beth** (ratio 0.800) son puentes frágiles, mientras que **chen hajaj** (ratio 0.803, betweenness 0.0036) y **James Lauderdale Jr** (ratio 0.788, betweenness 0.0035) son los puentes más sólidos estructuralmente.

### 2.9 Relevancia para detección de anomalías

El análisis del grafo de miembros consolida los siguientes candidatos a anomalía: los **13 nodos hoja** y los **922 nodos con clustering < 0.5** como extremos estructurales; los **97 super-conectores** y casos como Pablo — alta betweenness con degree moderada — como roles estructurales atípicos; las **microcomunidades C2 y C21** como anomalías a nivel de subgrafo; las **74 aristas con peso > 5** como candidatas a anomalía de arista; los **487 nodos con ratio inter-comunidad > 0.5** como puentes estructurales; y la entidad **"GEEK by AKEIN Engineering"** como anomalía de naturaleza del nodo.

---

## 3. Grafo de Grupos

### 3.1 Estadísticas básicas y métricas globales

El grafo de grupos contiene **456 nodos y 6.692 aristas**. De los 602 grupos de `meta_groups`, solo el **75.7% aparece en el grafo** — los 146 grupos restantes (24.3%) no comparten ningún miembro con otros grupos y quedan completamente aislados. El grafo es **completamente conexo**.

La **densidad es de 0.0645**, casi cuatro veces superior a la del grafo de miembros (0.0182), lo que indica que los grupos comparten miembros entre sí con más frecuencia de lo que los miembros se conectan entre ellos. El **grado medio es de 29.35** con un rango de 1 a 182. El **coeficiente de clustering medio es de 0.5488**, significativamente inferior al del grafo de miembros (0.885), lo que refleja que los grupos forman comunidades menos cohesionadas — hay más diversidad en las conexiones entre grupos de distintas categorías temáticas.

### 3.2 Distribución de grado

La distribución presenta asimetría positiva, con media (29) superior a la mediana (18) y desviación típica (32) mayor que la media. La segmentación revela: 28 grupos hoja (grado = 1), 206 grupos de grado bajo (2-18), 109 de grado medio (19-42), 90 de grado alto (43-100) y 23 super-conectores con grado superior a 100. Los **28 grupos hoja** son proporcionalmente más que en el grafo de miembros (13) e incluyen grupos muy específicos como Nashville ColdFusion User Group o Nashville League of Legends.

### 3.3 Análisis de nodos extremos

**Stepping Out Social Dance Meetup** es el grupo más conectado con grado 182, a pesar de tener solo 1.778 miembros. En contraste, **Nashville Hiking Meetup**, con 15.838 miembros, tiene un grado de solo 136. Este contraste ilustra que el tamaño de un grupo no determina su conectividad en la red — los grupos sociales generalistas son mejores conectores que los grupos de actividades masivas específicas.

### 3.4 Métricas de centralidad

A diferencia del grafo de miembros, en el grafo de grupos existe una **correlación positiva clara entre degree y betweenness** — los grupos más conectados son también los que más actúan como puentes. **Stepping Out Social Dance Meetup** lidera en ambas métricas (degree centrality 0.400, betweenness 0.073). Es llamativo que **Sunday Assembly Nashville** (Religion & Beliefs) aparezca en el top 10 de ambas — una comunidad religiosa actuando como puente entre mundos temáticos muy distintos. Solo **89 grupos (19.5%) tienen betweenness = 0**, frente al 72.5% en el grafo de miembros — la red de grupos es estructuralmente más democrática.

### 3.5 Coeficiente de clustering local

La distribución del clustering local es muy distinta a la del grafo de miembros: en lugar de una distribución bimodal concentrada en 0 y 1, aquí es **más uniforme**, con grupos en todos los rangos. Hay **61 grupos con clustering = 1.0** y **33 con clustering < 0.1**, entre los que destaca Nashville Children in Nature (654 miembros) con clustering = 0 — un grupo grande pero completamente aislado estructuralmente.

### 3.6 Análisis de pesos de aristas

Las aristas representan el número de miembros compartidos entre dos grupos. La distribución es más rica que en el grafo de miembros: media de 2.30 (vs. 1.05 en miembros), rango de 1 a 91 (vs. máximo de 9 en miembros) y solo el 66.2% de las aristas con peso = 1 (vs. 95.9% en miembros).

El análisis de las aristas más pesadas revela dos patrones muy claros. El primero es el **hub tecnológico de NashJS**: aparece en 6 de las 10 aristas más fuertes, actuando como núcleo central de la comunidad Tech — Code for Nashville, NashReact, PyNash, Nashville .NET, Nashville UX y Nashville Software Beginners convergen en él. El segundo es el **vínculo inter-categoría más fuerte de toda la red**: Stepping Out Social Dance Meetup ↔ Middle TN 40+ singles con peso 91, conectando Dancing con Singles — categorías temáticamente distintas pero sociodemográficamente muy similares.

### 3.7 Correlación num_members vs. grado

La correlación entre el número de miembros registrados y el grado en la red es **moderada-alta pero no perfecta** (Pearson r=0.548, Spearman r=0.666). El scatter confirma dos tipos de anomalías: grupos **grandes pero poco conectados** como Nashville Hiking Meetup (15.838 miembros, grado 136), cuyos miembros son relativamente exclusivos; y grupos **pequeños pero muy conectados** como Stepping Out Social Dance Meetup (1.778 miembros, grado 182), cuyos miembros se distribuyen por una gran variedad de otros grupos.

### 3.8 Detección de comunidades

Louvain obtuvo una modularidad de **0.4427** frente a los 0.3729 de Greedy, detectando 6 comunidades frente a las 9 de Greedy. La modularidad es sensiblemente inferior a la del grafo de miembros (0.6755), confirmando que las comunidades de grupos están menos bien definidas.

| Métrica | Louvain | Greedy |
|---|---|---|
| Nº comunidades | 6 | 9 |
| Modularidad | 0.4427 | 0.3729 |
| Tamaño medio | 76.0 | 50.7 |
| Comunidad más grande | 194 | 263 |
| Comunidades < 10 nodos | 2 | 6 |

Las 6 comunidades tienen temáticas dominantes bien definidas. **C1 (Socializing, 194 nodos)** es la más grande y heterogénea. **C5 (Tech, 87 nodos)** es la más pura temáticamente y presenta el mayor grado medio (0.096). Las anomalías más llamativas son **C0 (2 nodos)** y **C3 (4 nodos, Religion & Beliefs)** — microcomunidades prácticamente aisladas, siendo C3 especialmente peculiar por tener "Christianity & Transhumanism" como grupo más relevante.

### 3.9 Análisis de nodos puente entre comunidades

El ratio inter-comunidad medio del grafo de grupos es **0.297**, significativamente superior al del grafo de miembros (0.181), coherente con su menor modularidad. El hallazgo más llamativo es que **8 de los 10 grupos con mayor ratio inter-comunidad pertenecen a la comunidad C2** (New Age & Spirituality), que actúa como principal exportadora de conexiones hacia otras comunidades. Grupos como Lesbians in the Workplace (LGBT, ratio 0.842), Nashvegans! (Movements & Politics, 0.753) o Intellectual Society of Greater Nashville (Education & Learning, 0.746) tienen temáticas transversales que atraen a miembros de comunidades muy distintas.

### 3.10 Relevancia para detección de anomalías

Los candidatos a anomalía del grafo de grupos son: los **146 grupos ausentes del grafo** como los más aislados del ecosistema; los **28 grupos hoja** con conectividad mínima; los grupos con **desajuste entre tamaño y grado** — Nashville Hiking Meetup y Stepping Out Social Dance en extremos opuestos; las **microcomunidades C0 y C3** como anomalías a nivel de subgrafo; **NashJS** como hub tecnológico extremo; y el vínculo **Stepping Out ↔ Middle TN 40+ singles** como arista de peso anómalamente alto.

---

## 4. Grafo Bipartito Miembro-Grupo

### 4.1 Estadísticas básicas y componentes conexas

El grafo bipartito contiene **25.233 nodos en total** — 24.631 nodos de miembro y 602 nodos de grupo — conectados por **45.583 aristas**. Es el grafo más completo del dataset. La **densidad es de 0.0031**. A diferencia de los otros dos grafos, el **grafo bipartito no es conexo** — presenta **25 componentes conexas**. Sin embargo, la componente principal concentra **25.171 de los 25.233 nodos (99.8%)**, por lo que la desconexión es marginal.

Las 24 componentes pequeñas tienen entre 2 y 7 nodos. Todas tienen exactamente 1 grupo, confirmando que son grupos cuyos miembros no tienen ninguna co-membresía con el resto de la red. Las 19 componentes de 2 nodos representan el caso más extremo: 1 miembro + 1 grupo completamente aislados. Destacan casos llamativos como **Cincinnati Conversation about Moments of Grace** — cuyo nombre incluye una ciudad fuera de Tennessee — y grupos muy específicos como witchcraft united o Das Squad. Todos estos grupos y sus miembros son candidatos directos a anomalía estructural.

### 4.2 Distribución de grado por tipo de nodo

**Nodos miembro.** La distribución es extremadamente concentrada: mediana = 1 y media = 1.85. El **66.7% de los miembros tienen grado 1** — solo están en un grupo. Solo **212 miembros (0.9%) pertenecen a más de 10 grupos** y el máximo es de 42 grupos para un único miembro.

**Nodos grupo.** La distribución es mucho más dispersa, con mediana de 24 miembros activos y media de 75.72. El **6.1% de los grupos (37) tienen un único miembro activo registrado**. El grupo más grande en el bipartito tiene 951 miembros activos.

### 4.3 Análisis de nodos extremos

**Shalini** lidera el ranking de miembros con 42 grupos, seguida de Matt Kenigson (34) y Taylor Michael Matson (32). Los top 10 coinciden exactamente con los super-conectores del grafo de miembros, confirmando la coherencia entre grafos.

En cuanto a grupos, **20s in Nashville** lidera con 951 miembros activos, seguido de Nashville Hiking Meetup (878) y NashJS (760). Es relevante que estos valores son muy inferiores a los de `meta_groups` — Nashville Hiking Meetup tiene 15.838 miembros registrados pero solo 878 activos — lo que anticipa el hallazgo del ratio de actividad.

### 4.4 Comparativa miembros registrados vs. activos

El cruce entre miembros registrados en `meta_groups` y miembros con actividad real en el bipartito es uno de los hallazgos más importantes del EDA completo. El **ratio de actividad medio es de 0.172** (mediana 0.139) — de media, solo el **17% de los miembros registrados tienen actividad real** registrada.

**226 grupos (37.5%) tienen un ratio inferior al 10%**. Los grupos con mayor ratio son mayoritariamente **grupos tecnológicos pequeños y especializados**: T-VOE: Tennessee Voice Over Exchange (88.9%), Nashville Blockchain Meetup (61.7%), Nashville DevOps Meetup (57.9%). En el extremo opuesto destacan los **grupos fantasma**: Nashville Social Crew (0.07%, 4.017 miembros registrados y solo 3 activos), Nashville Co-Ed Kickball (0.16%, 620 miembros y 1 activo) o Hiking Club of Nashville (0.62%, 3.381 miembros y 21 activos).

### 4.5 Análisis de pesos de aristas

El peso representa el número de eventos del grupo a los que ha asistido el miembro — la única métrica de **actividad real** del dataset. La distribución es muy concentrada: mediana = 1, media = 2.78 y **60.5% de aristas con peso = 1**. Solo **156 aristas (0.3%) tienen peso superior a 50** y **49 (0.1%) superior a 100**. El peso máximo es 200, valor que corresponde al truncamiento de la API de Meetup ya identificado en `meta_events`. Los grupos con mayor fidelización pertenecen predominantemente a **New Age & Spirituality, Religion & Beliefs y Career & Business**.

### 4.6 Análisis de actividad por miembro

El **44% de los miembros tienen total_events = 1** — casi la mitad del dataset ha asistido a un único evento en toda su historia. La mediana de eventos totales es solo 2. Los miembros más activos presentan dos perfiles claramente diferenciados: **Becki Baumgartner** (513 eventos, 8 grupos, media 64 eventos/grupo) representa el perfil fiel y comprometido, mientras que **Taylor Michael Matson** (201 eventos, 32 grupos, media 6.3 eventos/grupo) representa el perfil explorador.

### 4.7 Perfiles de comportamiento

La segmentación de miembros por perfil de actividad identifica cinco grupos con implicaciones directas para la detección de anomalías:

**Normal (54.6%, 13.444 miembros):** actividad moderada y variada, comportamiento base de la red.

**Inactivo (44.0%, 10.834 miembros):** solo han asistido a un único evento. Constituyen la anomalía más masiva del dataset.

**Explorador (0.6%, 151 miembros):** pertenecen a más de 10 grupos con media de eventos por grupo inferior a 3. Perfil susceptible de representar comportamiento artificial o cuentas de baja calidad.

**Fiel (0.6%, 141 miembros):** pertenecen a 3 o menos grupos con media superior a 30 eventos por grupo. El perfil más valioso para los organizadores.

**Hiperactivo (0.2%, 61 miembros):** más de 100 eventos totales. Merecen inspección individual.

### 4.8 Redundancy coefficient

El redundancy coefficient es el equivalente bipartito del coeficiente de clustering. Solo es calculable para miembros con grado >= 2, lo que excluye a los 16.422 miembros con un único grupo. De los **8.209 miembros aptos**, la distribución está **fuertemente concentrada en 1.0**: mediana = 1.0, media = 0.835 y **66.7% con redundancy = 1.0**. Los **692 miembros con redundancy = 0 (8.4%)** son estructuralmente los más relevantes: pertenecen a múltiples grupos que no comparten ningún miembro común entre sí, actuando como puentes entre comunidades completamente distintas en el bipartito.

### 4.9 Relevancia para detección de anomalías

El grafo bipartito aporta una dimensión de análisis única: la **actividad real** de los miembros. Los candidatos a anomalía se organizan en tres niveles:

A nivel de **comportamiento individual**: los 10.834 miembros inactivos (44%), los 151 exploradores y los 61 hiperactivos constituyen perfiles de actividad atípicos.

A nivel de **grupos**: los 226 grupos con ratio de actividad inferior al 10% y los grupos fantasma como Nashville Social Crew (ratio 0.07%) representan anomalías de participación muy claras.

A nivel **estructural**: las 24 componentes aisladas, los 692 miembros con redundancy = 0 y grado >= 2, y los 37 grupos con un único miembro activo son los candidatos estructurales más sólidos.