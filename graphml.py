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


# --- Bipartite Member-Group graph ---
member_group_edges = dataframes["member-to-group-edges"].copy()

# Prefijar IDs para evitar colisiones entre espacios de member_id y group_id
member_group_edges['member_id'] = 'member_' + member_group_edges['member_id'].astype(str)
member_group_edges['group_id']  = 'group_'  + member_group_edges['group_id'].astype(str)

MG = nx.from_pandas_edgelist(
    member_group_edges,
    source='member_id',
    target='group_id',
    edge_attr='weight'
)

# Marcar nodos con su tipo (necesario para algoritmos bipartitos de NetworkX)
member_nodes = [n for n in MG.nodes() if n.startswith('member_')]
group_nodes  = [n for n in MG.nodes() if n.startswith('group_')]

nx.set_node_attributes(MG, {n: 0 for n in member_nodes}, name='bipartite')
nx.set_node_attributes(MG, {n: 1 for n in group_nodes},  name='bipartite')

# Añadir atributos de meta_members (con IDs prefijados)
meta_members_attrs = dataframes["meta-members"].copy()
meta_members_attrs['member_id'] = 'member_' + meta_members_attrs['member_id'].astype(str)
nx.set_node_attributes(MG, meta_members_attrs.set_index('member_id').to_dict('index'))

# Añadir atributos de meta_groups (con IDs prefijados)
meta_groups_attrs = dataframes["meta-groups"].copy()
meta_groups_attrs['group_id'] = 'group_' + meta_groups_attrs['group_id'].astype(str)
nx.set_node_attributes(MG, meta_groups_attrs.set_index('group_id').to_dict('index'))


print("--- Graphs Created ---")


print('\n--- Saving Graphs ---')
os.makedirs("./graphs/graphml/", exist_ok=True)

nx.write_graphml(G,  "./graphs/graphml/group_graph.graphml")
nx.write_graphml(M,  "./graphs/graphml/member_graph.graphml")
nx.write_graphml(MG, "./graphs/graphml/member_to_group_graph.graphml")

print("--- Graphs Saved Successfully ---")