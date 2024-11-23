import logging
from bluepysnap import Circuit
from neo4j_connector import Neo4jConnector
from .data_extraction import extract_nodes, extract_edges
from .neo4j_operations import (
    clear_database,
    create_nodegroup_nodes,
    create_nodegroup_relationships,
    bulk_insert_neuron_nodes,
    bulk_insert_edges,
    create_neuron_belongs_to_nodegroup_relationships,
)

logger = logging.getLogger(__name__)


def load_circuit(
    circuit_config_path: str,
    connector: Neo4jConnector,
    node_proportion: float,
    edge_proportion: float,
    node_set: str,
):
    # Initialize BluePySnap Circuit
    circuit = Circuit(circuit_config_path)

    # Clear the database for a fresh start
    clear_database(connector)

    # Extract nodes and edges
    sampled_nodes_df, belongs_to_relations, populations = extract_nodes(
        circuit, node_set, node_proportion
    )
    edges = extract_edges(circuit, edge_proportion, sampled_nodes_df)

    # Convert sampled_nodes_df to list of dicts
    nodes = sampled_nodes_df.to_dict("records")

    # Create NodeGroup nodes based on mtype
    create_nodegroup_nodes(connector, "mtype", "MType", nodes)
    create_nodegroup_nodes(connector, "synapse_class", "SClass", nodes)
    # Insert neuron nodes with mtype labels in chunks
    chunk_size = 1000
    for i in range(0, len(nodes), chunk_size):
        chunk = nodes[i : i + chunk_size]
        bulk_insert_neuron_nodes(connector, chunk)

    # Create BELONGS_TO_MTYPE relationships
    create_neuron_belongs_to_nodegroup_relationships(connector, "mtype", "MType", nodes)
    # Insert edges in chunks
    for i in range(0, len(edges), chunk_size):
        chunk = edges[i : i + chunk_size]
        bulk_insert_edges(connector, chunk)

    # Create relationships between NodeGroup nodes
    create_nodegroup_relationships(connector, "mtype")
    create_nodegroup_relationships(connector, "synapse_class")
