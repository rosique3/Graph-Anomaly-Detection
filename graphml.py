import os
import kagglehub
import pandas as pd
import networkx as nx



path = kagglehub.dataset_download("stkbailey/nashville-meetup")

archivos = os.listdir(path)

dataframes = {} 

print("--- Loading Files ---")

for archivo in archivos:

    df = pd.read_csv(os.path.join(path, archivo))
    
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
        
    nombre_clave = archivo.replace(".csv", "")
    
    dataframes[nombre_clave] = df
    
    print(f"✅ Loaded: {nombre_clave}")
    print("-" * 20)

print("--- Load Completed ---")
print(f"Dictionary of DataFrames created with the following keys: {dataframes.keys()}")


print('\n--- Creating Graphs ---')
# Group graph
G = nx.from_pandas_edgelist(dataframes["group-edges"], 
                            source='group1', 
                            target='group2', 
                            edge_attr='weight')

nx.set_node_attributes(G,
                       dataframes["meta-groups"].set_index("group_id").to_dict("index"))


# Member graph
M = nx.from_pandas_edgelist(dataframes["member-edges"], 
                            source='member1', 
                            target='member2', 
                            edge_attr='weight')

nx.set_node_attributes(M,
                       dataframes["meta-members"].set_index("member_id").to_dict("index"))



# Event graph
MG = nx.from_pandas_edgelist(dataframes["member-to-group-edges"], 
                            source='member_id', 
                            target='group_id', 
                            edge_attr='weight')

nx.set_node_attributes(MG,
                       dataframes["meta-groups"].set_index("group_id").to_dict("index"))
nx.set_node_attributes(MG,
                       dataframes["meta-members"].set_index("member_id").to_dict("index"))


# Global graph
GF = nx.Graph()

GF.add_weighted_edges_from(
    dataframes["group-edges"][["group1", "group2", "weight"]].values
)

GF.add_weighted_edges_from(
    dataframes["member-edges"][["member1", "member2", "weight"]].values
)

GF.add_weighted_edges_from(
    dataframes["member-to-group-edges"][["member_id", "group_id", "weight"]].values
)

nx.set_node_attributes(GF,
                       dataframes["meta-groups"].set_index("group_id").to_dict("index"))
nx.set_node_attributes(GF,
                       dataframes["meta-members"].set_index("member_id").to_dict("index"))

print("--- Graphs Created ---")



print('\n--- Saving Graphs ---')
# Save Graphs
nx.write_graphml(G, "./graphs/graphml/group_graph.graphml")
nx.write_graphml(M, "./graphs/graphml/member_graph.graphml")
nx.write_graphml(MG, "./graphs/graphml/member_to_group_graph.graphml")
nx.write_graphml(GF, "./graphs/graphml/global_graph.graphml")

print("--- Graphs Saved ---")