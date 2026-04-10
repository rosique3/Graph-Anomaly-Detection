import os
import pandas as pd
import networkx as nx

# Configuración de la ruta local
DATA_PATH = 'data/processed/'

# Listamos los archivos en la carpeta procesada
archivos = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]

dataframes = {} 

print("--- Loading Files from Local Storage ---")

for archivo in archivos:
    # Carga del CSV
    df = pd.read_csv(os.path.join(DATA_PATH, archivo))
        
    # Usamos el nombre del archivo sin extensión como clave del diccionario
    nombre_clave = archivo.replace(".csv", "")
    
    dataframes[nombre_clave] = df
    
    print(f"✅ Loaded: {nombre_clave}")
    print("-" * 20)

print("--- Load Completed ---")
print(f"Dictionary keys: {list(dataframes.keys())}")


print('\n--- Creating Graphs ---')

# --- Group graph ---
G = nx.from_pandas_edgelist(dataframes["group-edges"], 
                            source='group1', 
                            target='group2', 
                            edge_attr='weight')

nx.set_node_attributes(G,
                       dataframes["meta-groups"].set_index("group_id").to_dict("index"))


# --- Member graph ---
M = nx.from_pandas_edgelist(dataframes["member-edges"], 
                            source='member1', 
                            target='member2', 
                            edge_attr='weight')

nx.set_node_attributes(M,
                       dataframes["meta-members"].set_index("member_id").to_dict("index"))


# --- Member-to-Group graph ---
MG = nx.from_pandas_edgelist(dataframes["member-to-group-edges"], 
                            source='member_id', 
                            target='group_id', 
                            edge_attr='weight')

nx.set_node_attributes(MG, dataframes["meta-groups"].set_index("group_id").to_dict("index"))
nx.set_node_attributes(MG, dataframes["meta-members"].set_index("member_id").to_dict("index"))


# --- Global graph ---
GF = nx.Graph()

# Añadimos aristas de las tres fuentes
GF.add_weighted_edges_from(dataframes["group-edges"][["group1", "group2", "weight"]].values)
GF.add_weighted_edges_from(dataframes["member-edges"][["member1", "member2", "weight"]].values)
GF.add_weighted_edges_from(dataframes["member-to-group-edges"][["member_id", "group_id", "weight"]].values)

# Añadimos metadatos a los nodos
nx.set_node_attributes(GF, dataframes["meta-groups"].set_index("group_id").to_dict("index"))
nx.set_node_attributes(GF, dataframes["meta-members"].set_index("member_id").to_dict("index"))

print("--- Graphs Created ---")


print('\n--- Saving Graphs ---')
# Asegúrate de que la carpeta de destino exista
os.makedirs("./graphs/graphml/", exist_ok=True)

nx.write_graphml(G, "./graphs/graphml/group_graph.graphml")
nx.write_graphml(M, "./graphs/graphml/member_graph.graphml")
nx.write_graphml(MG, "./graphs/graphml/member_to_group_graph.graphml")
nx.write_graphml(GF, "./graphs/graphml/global_graph.graphml")

print("--- Graphs Saved Successfully ---")