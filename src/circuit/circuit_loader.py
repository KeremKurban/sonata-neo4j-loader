import argparse
import os
import logging
from bluepysnap import Circuit
from dotenv import load_dotenv
from neo4j_connector import Neo4jConnector
from sonata_to_neo4j.src.circuit.neo4j_operations import (
    clear_database,
    create_nodegroup_nodes,
    create_nodegroup_relationships,
    bulk_insert_neuron_nodes,
    bulk_insert_edges,
    create_neuron_belongs_to_nodegroup_relationships,
)
from sonata_to_neo4j.src.circuit.data_extraction import extract_nodes, extract_edges

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(circuit_config_path: str, apply_labels: bool = False) -> None:
    """
    Main function to process SONATA files, extract nodes, edges, and populations, and insert them into Neo4j.

    Parameters
    ----------
    circuit_config_path : str
        Path to the circuit configuration JSON file.
    apply_labels : bool, optional
        Whether to apply labels to nodes based on their 'mtype' property.
    """
    # Load environment variables
    load_dotenv()

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    node_proportion = float(os.getenv("NODE_PROPORTION", 1.0))
    edge_proportion = float(os.getenv("EDGE_PROPORTION", 1.0))
    node_set = os.getenv("NODE_SET")
    # apply_mtype_labels = os.getenv("APPLY_MTYPE_LABELS")
    # Initialize BluePySnap Circuit
    circuit = Circuit(circuit_config_path)

    # Connect to Neo4j
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)

    # Clear the database for a fresh start
    clear_database(connector)

    # Extract nodes and edges using functions above
    sampled_nodes_df, belongs_to_relations, populations = extract_nodes(
        circuit, node_set, node_proportion
    )
    edges = extract_edges(circuit, edge_proportion, sampled_nodes_df)

    # Convert sampled_nodes_df to list of dicts
    nodes = sampled_nodes_df.to_dict("records")

    # Create NodeGroup nodes based on mtype
    create_nodegroup_nodes(connector, "mtype", "MType", nodes)

    # Insert population nodes
    # bulk_insert_population_nodes(connector, populations)

    # Insert neuron nodes with mtype labels in chunks
    chunk_size = 1000
    for i in range(0, len(nodes), chunk_size):
        chunk = nodes[i : i + chunk_size]
        bulk_insert_neuron_nodes(connector, chunk)

    # Create BELONGS_TO_MTYPE relationships
    create_neuron_belongs_to_nodegroup_relationships(connector, "mtype", "MType", nodes)

    # # Insert BELONGS_TO relationships in chunks
    # for i in range(0, len(belongs_to_relations), chunk_size):
    #     chunk = belongs_to_relations[i : i + chunk_size]
    #     bulk_insert_belongs_to_relationships(connector, chunk)

    # Insert edges in chunks
    for i in range(0, len(edges), chunk_size):
        chunk = edges[i : i + chunk_size]
        bulk_insert_edges(connector, chunk)

    # Create relationships between NodeGroup nodes
    create_nodegroup_relationships(connector)

    # Optionally add labels based on mtype
    # add_labels_based_on_mtype(connector, apply_mtype_labels)

    # Close connection
    connector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process SONATA files and insert data into Neo4j."
    )
    parser.add_argument(
        "--circuit_config",
        type=str,
        required=True,
        help="Path to the circuit configuration JSON file.",
    )

    args = parser.parse_args()

    main(args.circuit_config)
