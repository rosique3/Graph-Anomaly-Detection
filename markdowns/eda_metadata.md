# EDA Tabular — Meetup Tennessee

## 1. meta_members

### 1.1 Estadísticas básicas

El dataset contiene **24.591 miembros únicos** distribuidos en **893 ciudades distintas** y registrados en **64 "estados"** diferentes, aunque como veremos, no todos son estados válidos de EE.UU.

### 1.2 Distribución geográfica

La distribución geográfica es muy concentrada: el **91.7% de los miembros reside en Tennessee**, y dentro de este, **Nashville domina con el 60.1%** del total (14.779 miembros). El resto de ciudades con presencia significativa — Franklin, Brentwood, Murfreesboro, Hendersonville — pertenecen todas al área metropolitana de Nashville, lo cual es coherente con el origen del dataset.

El **8.3% restante** (2.031 miembros) está fuera de Tennessee. Los estados más representados son los vecinos geográficos (Kentucky, Georgia, Alabama), lo que tiene sentido geográficamente. Sin embargo, la presencia de miembros de California, Nueva York o Illinois es más llamativa y podría responder a usuarios que se mudaron o que simplemente se apuntaron a grupos online.

La visualización geográfica sobre mapa mundial confirma este patrón de forma muy clara. Los puntos rojos — miembros con estado informado — se concentran de forma casi exclusiva en el sureste de EE.UU., con Nashville como epicentro visible. Los puntos morados — miembros sin estado, es decir internacionales — aparecen dispersos por Europa, Asia, África y América Latina, confirmando que su ausencia de estado no es un error sino una limitación del campo para usuarios fuera de EE.UU. La distribución global de estos puntos morados es coherente con los perfiles identificados en el muestreo cualitativo: Hong Kong, Lima, Málaga, Kumasi y otras ciudades de distintos continentes.

### 1.3 Calidad de datos

Se identificaron tres issues de calidad relevantes.

El primero son los **94 miembros sin valor en `state`**. Un muestreo cualitativo reveló que no se trata de datos corruptos, sino de **miembros internacionales** para los que la plataforma Meetup simplemente no registra estado al no ser un campo aplicable fuera de EE.UU. y Canadá. El mapa mundial confirma esta hipótesis visualmente — todos los puntos morados tienen coordenadas geográficamente coherentes con sus ciudades declaradas.

El segundo son los **20 miembros con códigos de estado inválidos** (valores como "C3", "F8", "17"...). Estos sí representan datos corruptos o mal formateados en la plataforma original, y se guardan como candidatos a anomalías de calidad de datos.

El tercero y más relevante es el de las **coordenadas de centroide**: el **28.9% de los miembros (7.107)** tiene asignadas las coordenadas del centro de su ciudad en lugar de su ubicación real. Esto ocurre cuando el usuario no especifica una dirección concreta y la plataforma resuelve la ubicación a nivel de ciudad. Como consecuencia, las coordenadas no son fiables a nivel individual y cualquier análisis espacial fino debe tratarse con cautela. Esta limitación es visible en el mapa: la alta densidad de puntos superpuestos en la zona de Nashville responde en parte a este fenómeno de centroide, no solo a la concentración real de miembros.

### 1.4 Relevancia para detección de anomalías

Los miembros internacionales, los de fuera de Tennessee y los que presentan datos de estado corruptos constituyen un primer conjunto de candidatos a anomalía, no necesariamente por comportamiento anómalo en el grafo, sino por su naturaleza atípica dentro de una red fundamentalmente local. Será interesante contrastar en fases posteriores si estos perfiles también presentan patrones de conexión inusuales.

---

## 2. meta_groups

### 2.1 Estadísticas básicas

El dataset contiene **602 grupos únicos** distribuidos en **31 categorías temáticas**, organizados por **522 organizadores distintos**. La ratio de casi un organizador por grupo indica que la mayoría de los grupos tienen un gestor dedicado, aunque como veremos, hay excepciones relevantes.

### 2.2 Distribución por categoría

La distribución de grupos por categoría está claramente sesgada hacia dos temáticas dominantes: **Tech** (100 grupos, 16.6%) y **Career & Business** (93 grupos, 15.5%), que juntas concentran casi un tercio del total. En el extremo opuesto, categorías como Cars & Motorcycles (2 grupos), Singles (3) o Writing (3) tienen una presencia testimonial.

Sin embargo, el número de grupos por categoría no refleja fielmente su peso real en la red. Al analizar el **total de miembros acumulados por categoría**, emerge un patrón distinto: **Outdoors & Adventure lidera con 59.445 miembros totales** a pesar de tener solo 36 grupos, superando a Tech (46.227) y Career & Business (44.302). Esto revela que sus grupos son de media mucho más grandes, como confirma su media de 1.651 miembros por grupo frente a los 462 de Tech.

El análisis por categoría permite identificar cuatro perfiles claramente diferenciados. En el cuadrante de pocos grupos y media alta destacan Outdoors & Adventure, Singles, Writing y Dancing — categorías de nicho popular con grupos muy concurridos. En el cuadrante de pocos grupos y media baja se sitúan Support, Hobbies & Crafts y Parents & Family, con comunidades pequeñas y homogéneas. Tech y Career & Business ocupan el cuadrante de muchos grupos y media moderada, acumulando volumen total gracias a su cantidad. Socializing ocupa una posición central equilibrada con un volumen total considerable.

### 2.3 Distribución de num_members

La distribución de `num_members` presenta una **fuerte asimetría positiva** característica de redes sociales, con una media de 551 miembros muy por encima de la mediana de 201. El rango es extremadamente amplio, desde grupos con apenas 2 miembros hasta el grupo más grande con **15.838 miembros** (Nashville Hiking Meetup).

Aplicando el criterio IQR (umbral superior: 1.400 miembros), se identifican **68 grupos outlier** (11.3% del total), entre los que destacan Nashville Hiking Meetup (15.838), Paddle Adventures Unlimited (6.331) y Eat Love Nash (5.008). En el extremo opuesto, **13 grupos tienen menos de 10 miembros**, correspondiendo en su mayoría a grupos de Support, Career & Business y categorías muy específicas, que probablemente sean grupos recién creados o abandonados.

La comparación entre media y mediana por categoría revela qué categorías están dominadas por uno o pocos grupos gigantes. Outdoors & Adventure presenta la mayor asimetría (media 1.651 vs. mediana 763), consecuencia directa del peso de Nashville Hiking Meetup. Photography muestra un patrón similar (media 971 vs. mediana 318) por la influencia de Nashville Social Crew (4.017 miembros).

### 2.4 Análisis de organizadores

De los 522 organizadores únicos, **457 (87.5%) gestionan un único grupo**, 51 gestionan dos y solo 14 organizan tres o más. El organizador más activo es **Micah Redding** con 4 grupos en 3 categorías distintas (Religion & Beliefs y Tech), seguido de trece organizadores con 3 grupos cada uno.

Atendiendo al volumen de miembros acumulados, destacan **Michael Gabelman** (5.334 miembros, Outdoors & Adventure y Socializing) y **Charles** (4.962 miembros, Music, Socializing y Career & Business) como los organizadores de mayor alcance. Ambos gestionan grupos en categorías temáticamente diversas, lo que anticipa un rol de nodo puente entre comunidades en el grafo.

Merece mención especial el perfil de **Jubilee Church**, una organización religiosa que organiza grupos de Language & Ethnic Identity, Food & Drink y Music, utilizando Meetup presumiblemente como herramienta de dinamización comunitaria. Igualmente llamativo es el caso del organizador `221183678` (sin perfil de miembro registrado), responsable de tres grupos **#Resist** distribuidos geográficamente por Brentwood, Antioch y Franklin — un patrón de activismo político coordinado que resulta estructuralmente relevante para la detección de anomalías.

### 2.5 Calidad de datos e integridad referencial

Se identificaron **29 organizadores (5.6%)** cuyos IDs no tienen correspondencia en `meta_members`, afectando a **35 grupos**. Entre los grupos afectados se encuentran algunos de los más relevantes del dataset, como PyNash (1.442 miembros) y Nashville Social Crew (4.017 miembros), lo que sugiere que estos grupos podrían estar gestionados por entidades o cuentas corporativas en lugar de miembros individuales. Este hallazgo constituye una limitación de integridad referencial del dataset que debe tenerse en cuenta en fases posteriores del análisis.

### 2.6 Relevancia para detección de anomalías

El análisis de `meta_groups` arroja varios candidatos a anomalía estructural. Los **68 grupos outlier por num_members** representan nodos de grado potencialmente muy alto en el grafo bipartito. Los **organizadores multi-categoría** como Micah Redding, Charles o Michael Gabelman serán previsiblemente nodos puente entre comunidades. Los **13 grupos con menos de 10 miembros** podrían corresponder a nodos aislados o periféricos en el grafo. Finalmente, los **patrones coordinados** como los grupos #Resist o los grupos de Jubilee Church representan estructuras de comunidad intencionalmente diseñadas, cuyo comportamiento en el grafo puede diferir significativamente del resto.

---

## 3. meta_events

### 3.1 Estadísticas básicas

El dataset contiene **19.307 eventos únicos** distribuidos entre los **602 grupos** ya identificados en `meta_groups`, lo que supone una media de 32 eventos por grupo. Sin embargo, la fuerte asimetría de la distribución (mediana de 12 eventos) indica que esta media está inflada por grupos muy activos, siendo lo más habitual grupos con menos de 25 eventos. El rango temporal abarca exactamente **729 días**, desde noviembre de 2015 hasta octubre de 2017, cubriendo dos años completos de actividad.

### 3.2 Distribución de eventos por grupo

La distribución de eventos por grupo presenta la misma **asimetría positiva** observada en `meta_groups` con `num_members`. La gran mayoría de grupos concentra pocos eventos, mientras una minoría acumula una actividad muy superior a la media.

Un hallazgo crítico es la presencia de **32 grupos (5.3%) con exactamente 200 eventos**, valor que corresponde al **límite de extracción de la API de Meetup** y no al número real de eventos celebrados. Estos grupos son precisamente los más activos y con mayor número de miembros del dataset — Nashville Hiking Meetup, Tennessee Hiking Group, Nashville Social Crew, entre otros — lo que significa que el truncamiento afecta de forma desproporcionada a los nodos más relevantes de la red. En total, **6.400 eventos (33.1% del dataset) provienen de grupos truncados**, lo cual constituye una limitación seria para cualquier análisis de actividad posterior.

En el extremo opuesto, **81 grupos tienen un único evento registrado**, correspondiendo probablemente a grupos recién creados o abandonados tras su primera actividad. La integridad referencial entre `meta_events` y `meta_groups` es perfecta: todos los `group_id` presentes en `meta_events` tienen correspondencia en `meta_groups`.

### 3.3 Análisis temporal

La serie temporal mensual muestra una **tendencia de crecimiento prácticamente monotónica** a lo largo de los dos años del dataset. El volumen de eventos pasa de los ~400 mensuales a finales de 2015 hasta superar los 1.400 en octubre de 2017, reflejando una comunidad en plena expansión durante el periodo de captura. Se aprecia una leve caída estacional en noviembre-diciembre de 2016, coherente con la reducción de actividad social típica del periodo navideño.

El **sábado es el día con mayor número de eventos** (más de 4.000), seguido del domingo, lo que indica que los meetups de Tennessee son mayoritariamente actividades de fin de semana. Entre semana, el patrón es relativamente uniforme de martes a jueves, con una caída notable el viernes y el mínimo absoluto el lunes.

### 3.4 Calidad de datos: distribución horaria

La distribución por hora del día revela un **problema de calidad relevante**: los picos en las horas 0 y 23 son los más altos de todo el día, lo cual es inconsistente con el comportamiento esperado de eventos sociales. El análisis detallado confirma que **2.525 eventos (13.1%) tienen hora 00:00:00** — probablemente eventos registrados sin hora específica a los que la plataforma asigna medianoche por defecto — y **2.262 eventos (11.7%) tienen hora 23:00:00**, valor que también aparece como posible hora por defecto en Meetup. Un tercer grupo minoritario de 97 eventos (0.5%) presenta la hora 23:15:00 como posible valor de relleno.

En total, el **25.3% de los eventos tienen una hora de inicio no fiable**. Una vez excluidos estos valores, la distribución horaria real muestra el patrón esperable: actividad concentrada entre las 13h y las 20h, con el pico en torno a las 15-17h.

### 3.5 Relevancia para detección de anomalías

El análisis de `meta_events` aporta dos dimensiones relevantes para la detección de anomalías. Por un lado, los **32 grupos truncados** representan nodos cuya actividad real es desconocida y superior a la registrada, lo que puede distorsionar métricas de centralidad y actividad en el grafo. Por otro lado, los **patrones temporales anómalos** — horas por defecto, grupos con un único evento — constituyen señales de baja calidad de datos que conviene filtrar o tratar antes de construir las features para los modelos de detección.