import argparse
import json
from bluepysnap import Circuit
from reader import read_nodes, read_edges
from neo4j_connector import Neo4jConnector
from subsampling import subsample_nodes, subsample_edges
from dotenv import load_dotenv
import os
import pandas as pd

def main(circuit_config_path: str):
    """
    Main function to read nodes and edges from SONATA files, subsample them,
    and insert them into a Neo4j database.
    """
    # Load environment variables
    load_dotenv()

    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USER')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    node_proportion = float(os.getenv('NODE_PROPORTION', 1.0))
    edge_proportion = float(os.getenv('EDGE_PROPORTION', 1.0))

    # Initialize BluePySnap Circuit
    circuit = Circuit(circuit_config_path)

    # Read and subsample nodes using BluePySnap
    nodes_df = []
    for pop_name in circuit.nodes.population_names:
        node_population = circuit.nodes[pop_name]
        node_df = node_population.get()
        nodes_df.append(node_df)

    # Concatenate all node DataFrames
    all_nodes_df = pd.concat(nodes_df, ignore_index=True)
    sampled_nodes_df = all_nodes_df.sample(frac=node_proportion)

    # Read and subsample edges using BluePySnap
    edges_df = []
    for pop_name in circuit.edges.population_names:
        edge_population = circuit.edges[pop_name]
        edge_df = edge_population.get()
        edges_df.append(edge_df)

    # Concatenate all edge DataFrames
    all_edges_df = pd.concat(edges_df, ignore_index=True)
    sampled_edges_df = all_edges_df.sample(frac=edge_proportion)

    # Connect to Neo4j and insert data
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)

    for _, row in sampled_nodes_df.iterrows():
        node = {
            'id': row['node_id'],  # Assuming 'node_id' is a column in the DataFrame
            'properties': row.to_dict()
        }
        connector.create_node(node['id'], node['properties'])

    for _, row in sampled_edges_df.iterrows():
        edge = {
            'start_id': row['source_node_id'],  # Assuming 'source_node_id' is a column
            'end_id': row['target_node_id'],    # Assuming 'target_node_id' is a column
            'properties': row.to_dict()
        }
        connector.create_edge(edge['start_id'], edge['end_id'], edge['properties'])

    connector.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SONATA files and insert data into Neo4j.")
    parser.add_argument('--circuit_config', type=str, required=True, help='Path to the circuit configuration JSON file.')

    args = parser.parse_args()

    main(args.circuit_config)