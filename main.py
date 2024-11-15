from reader import read_nodes, read_edges
from neo4j_connector import Neo4jConnector
from subsampling import subsample_nodes, subsample_edges
import argparse

def main(circuit_config: dict, neo4j_uri: str, neo4j_user: str, neo4j_password: str, node_sample_size: int, edge_sample_size: int):
    """
    Main function to read nodes and edges from SONATA files, subsample them,
    and insert them into a Neo4j database.

    Parameters:
    - circuit_config (dict): Configuration dictionary containing paths to the
      nodes and edges files.
    - neo4j_uri (str): URI for connecting to the Neo4j database.
    - neo4j_user (str): Username for Neo4j authentication.
    - neo4j_password (str): Password for Neo4j authentication.
    - node_sample_size (int): Number of nodes to sample from the nodes file.
    - edge_sample_size (int): Number of edges to sample from the edges file.
    """
    nodes = read_nodes(circuit_config['nodes_file'])
    edges = read_edges(circuit_config['edges_file'])

    sampled_nodes = subsample_nodes(nodes, node_sample_size)
    sampled_edges = subsample_edges(edges, edge_sample_size)

    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)

    for node in sampled_nodes:
        connector.create_node(node['id'], node['properties'])

    for edge in sampled_edges:
        connector.create_edge(edge['start_id'], edge['end_id'], edge['properties'])

    connector.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SONATA files and insert data into Neo4j.")
    parser.add_argument('--nodes_file', type=str, required=True, help='Path to the nodes file.')
    parser.add_argument('--edges_file', type=str, required=True, help='Path to the edges file.')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687', help='URI for connecting to the Neo4j database.')
    parser.add_argument('--neo4j_user', type=str, default='neo4j', help='Username for Neo4j authentication.')
    parser.add_argument('--neo4j_password', type=str, default='password', help='Password for Neo4j authentication.')
    parser.add_argument('--node_sample_size', type=int, required=True, help='Number of nodes to sample.')
    parser.add_argument('--edge_sample_size', type=int, required=True, help='Number of edges to sample.')

    args = parser.parse_args()

    circuit_config = {
        'nodes_file': args.nodes_file,
        'edges_file': args.edges_file
    }
    main(circuit_config, args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.node_sample_size, args.edge_sample_size)