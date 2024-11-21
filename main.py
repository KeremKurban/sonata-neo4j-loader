import argparse
import os
import logging
from typing import Any, Dict, List, Tuple
import pandas as pd
from bluepysnap import Circuit
from dotenv import load_dotenv
from neo4j_connector import Neo4jConnector
import libsonata
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_constraints_and_indices(connector: Neo4jConnector) -> None:
    """
    Add constraints and indices to the Neo4j database for efficient imports and querying.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    """
    try:
        with connector.driver.session() as session:
            # Create uniqueness constraint for Neurons on (id, population_name)
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Neuron) REQUIRE (n.id, n.population_name) IS UNIQUE"
            )
            # Create uniqueness constraint for Populations
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Population) REQUIRE p.name IS UNIQUE"
            )
            logger.info("Constraints added successfully.")
    except Exception as e:
        logger.error(f"Error adding constraints: {e}")


def bulk_insert_population_nodes(
    connector: Neo4jConnector, populations: List[Dict[str, Any]]
) -> None:
    """
    Insert population nodes into Neo4j in bulk.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    populations : list of dict
        A list of dictionaries containing population properties.
    """
    query = """
    UNWIND $populations AS population
    MERGE (p:Population {name: population.name})
    SET p += population
    """
    try:
        with connector.driver.session() as session:
            session.run(query, populations=populations)
            logger.info("Population nodes inserted successfully.")
    except Exception as e:
        logger.error(f"Error inserting population nodes: {e}")


def bulk_insert_neuron_nodes(
    connector: Neo4jConnector, nodes: List[Dict[str, Any]]
) -> None:
    """
    Insert neuron nodes into Neo4j in bulk.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    nodes : list of dict
        A list of dictionaries containing neuron properties.
    """
    query = """
    UNWIND $nodes AS node
    MERGE (n:Neuron {id: node.id, population_name: node.population_name})
    SET n += node
    """
    try:
        with connector.driver.session() as session:
            session.run(query, nodes=nodes)
            logger.info("Neuron nodes inserted successfully.")
    except Exception as e:
        logger.error(f"Error inserting neuron nodes: {e}")


def bulk_insert_belongs_to_relationships(
    connector: Neo4jConnector, belongs_to_relations: List[Dict[str, Any]]
) -> None:
    """
    Create BELONGS_TO relationships between neurons and populations in bulk.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    belongs_to_relations : list of dict
        A list of dictionaries with 'node_id' and 'population_name' keys.
    """
    query = """
    UNWIND $relations AS rel
    MATCH (n:Neuron {id: rel.node_id, population_name: rel.population_name})
    MATCH (p:Population {name: rel.population_name})
    MERGE (n)-[:BELONGS_TO]->(p)
    """
    try:
        with connector.driver.session() as session:
            session.run(query, relations=belongs_to_relations)
            logger.info("BELONGS_TO relationships created successfully.")
    except Exception as e:
        logger.error(f"Error creating BELONGS_TO relationships: {e}")


def bulk_insert_edges(connector: Neo4jConnector, edges: List[Dict[str, Any]]) -> None:
    """
    Insert edges into Neo4j in bulk using UNWIND.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    edges : list of dict
        A list of dictionaries containing edge properties.
    """
    query = """
    UNWIND $edges AS edge
    MATCH (a:Neuron {id: edge.source_node_id, population_name: edge.source_population_name})
    MATCH (b:Neuron {id: edge.target_node_id, population_name: edge.target_population_name})
    CREATE (a)-[r:SYNAPSE]->(b)
    SET r += edge.properties
    """
    # Prepare edges by separating properties
    transformed_edges = [
        {
            "source_node_id": edge["source_node_id"],
            "source_population_name": edge["source_population_name"],
            "target_node_id": edge["target_node_id"],
            "target_population_name": edge["target_population_name"],
            "properties": {
                k: v
                for k, v in edge.items()
                if k
                not in [
                    "source_node_id",
                    "source_population_name",
                    "target_node_id",
                    "target_population_name",
                ]
            },
        }
        for edge in edges
    ]
    try:
        with connector.driver.session() as session:
            session.run(query, edges=transformed_edges)
            logger.info("Edges inserted successfully.")
    except Exception as e:
        logger.error(f"Error inserting edges: {e}")


def extract_nodes(
    circuit: Circuit, node_set: str, proportion: float
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract and subsample nodes from node populations.

    Parameters
    ----------
    circuit : Circuit
        An instance of the BluePySnap Circuit class.
    node_set: str
        Node set to be extracted from entire circuit.
    proportion : float
        The proportion of nodes to sample (between 0 and 1). If node set and proportion together is selected,
        joint probability will be retreived.

    Returns
    -------
    sampled_nodes_df : pd.DataFrame
        DataFrame containing sampled nodes with 'id' and 'population_name' columns.
    belongs_to_relations : list of dict
        A list of dictionaries for BELONGS_TO relationships.
    populations : list of dict
        A list of dictionaries containing population properties.
    """
    nodes_df_list = []
    belongs_to_relations = []
    populations = []
    for pop_name in circuit.nodes.population_names:
        logger.info(f"Extracting nodes from population: {pop_name}")
        node_population = circuit.nodes[pop_name]
        node_set_ids = node_population.ids(
            node_set
        ).tolist()  # can be non-contiguous ids
        node_storage = libsonata.NodeStorage(node_population.h5_filepath)
        population = node_storage.open_population(pop_name)
        # node_ids = list(range(population.size))
        selection = libsonata.Selection(node_set_ids)
        attr_types = list(population.attribute_names)
        attributes = {
            attr: population.get_attribute(attr, selection) for attr in attr_types
        }
        df = pd.DataFrame(attributes)
        df["id"] = node_set_ids
        df["population_name"] = pop_name
        nodes_df_list.append(df)
        belongs_to_relations.extend(
            [
                {"node_id": node_id, "population_name": pop_name}
                for node_id in node_set_ids
            ]
        )
        populations.append(
            {
                "name": pop_name,
                "size": population.size,
                "attr_types": attr_types,
            }
        )
    all_nodes_df = pd.concat(nodes_df_list, ignore_index=True)
    sampled_nodes_df = all_nodes_df.sample(frac=proportion, random_state=42)
    # Update belongs_to_relations to include only sampled nodes
    sampled_node_ids = set(
        zip(sampled_nodes_df["id"], sampled_nodes_df["population_name"])
    )
    belongs_to_relations = [
        rel
        for rel in belongs_to_relations
        if (rel["node_id"], rel["population_name"]) in sampled_node_ids
    ]
    return sampled_nodes_df, belongs_to_relations, populations


def extract_edges(
    circuit: Circuit, proportion: float, sampled_nodes_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Extract and subsample edges from edge populations.

    Parameters
    ----------
    circuit : Circuit
        An instance of the BluePySnap Circuit class.
    proportion : float
        The proportion of edges to sample (between 0 and 1).
    sampled_nodes_df : pd.DataFrame
        DataFrame containing sampled nodes with 'id' and 'population_name' columns.

    Returns
    -------
    edges : list of dict
        A list of dictionaries containing edge properties.
    """
    edges_df_list = []
    sampled_nodes_set = set(
        zip(sampled_nodes_df["id"], sampled_nodes_df["population_name"])
    )

    for pop_name in circuit.edges.population_names:
        logger.info(f"Extracting edges from population: {pop_name}")
        edge_population = circuit.edges[pop_name]
        edge_storage = libsonata.EdgeStorage(edge_population.h5_filepath)
        population = edge_storage.open_population(pop_name)

        # Select a subset of edge IDs based on the given proportion
        total_edges = population.size
        selected_edge_count = int(total_edges * proportion)
        edge_ids = list(range(total_edges))
        selected_edge_ids = random.sample(edge_ids, selected_edge_count)

        selection = libsonata.Selection(selected_edge_ids)
        attr_types = list(population.attribute_names)
        attributes = {
            attr: population.get_attribute(attr, selection) for attr in attr_types
        }
        source_node_ids = [population.source_node(eid) for eid in selected_edge_ids]
        target_node_ids = [population.target_node(eid) for eid in selected_edge_ids]

        # Parse pop_name to get source and target population names
        pop_name_parts = pop_name.split("__")
        if len(pop_name_parts) >= 3:
            source_population_name = pop_name_parts[0]
            target_population_name = pop_name_parts[1]
        elif pop_name == "default":
            logging.warning(
                "WARNING: Edge pop name doesnt comply with SONATA standard."
            )
            source_population_name = "hippocampus_neurons"
            target_population_name = "hippocampus_neurons"
        else:
            logger.error(f"Unable to parse population names from pop_name: {pop_name}")
            continue

        df = pd.DataFrame(attributes)
        df["source_node_id"] = source_node_ids
        df["source_population_name"] = source_population_name
        df["target_node_id"] = target_node_ids
        df["target_population_name"] = target_population_name

        # Vectorized filtering of edges where both nodes are sampled
        source_nodes = list(zip(df["source_node_id"], df["source_population_name"]))
        target_nodes = list(zip(df["target_node_id"], df["target_population_name"]))
        mask = pd.Series(source_nodes).isin(sampled_nodes_set) & pd.Series(
            target_nodes
        ).isin(sampled_nodes_set)
        df = df[mask]
        edges_df_list.append(df)

    if edges_df_list:
        all_edges_df = pd.concat(edges_df_list, ignore_index=True)
        edges = all_edges_df.to_dict("records")
    else:
        edges = []
    return edges


def add_labels_based_on_mtype(connector: Neo4jConnector, apply_labels: bool) -> None:
    """
    Optionally add labels to Neuron nodes based on their 'mtype' property.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    apply_labels : bool
        Whether to apply labels based on the 'mtype' property.
    """
    if not apply_labels:
        return

    query = """
    MATCH (n:Neuron)
    WHERE n.mtype IS NOT NULL
    CALL apoc.create.addLabels(n, [n.mtype]) YIELD node
    RETURN node.mtype AS mtype, labels(node) AS labels
    LIMIT 25
    """
    try:
        with connector.driver.session() as session:
            result = session.run(query)
            for record in result:
                logger.info(
                    f"Added label {record['mtype']} to node with labels {record['labels']}"
                )
    except Exception as e:
        logger.error(f"Error adding labels based on mtype: {e}")


def clear_database(connector: Neo4jConnector) -> None:
    """
    Clear all nodes, relationships, constraints, and indices in the Neo4j database.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    """
    try:
        with connector.driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("All nodes and relationships deleted successfully.")

            # Drop all constraints
            constraints = session.run("SHOW CONSTRAINTS")
            for constraint in constraints:
                constraint_name = constraint["name"]
                session.run(f"DROP CONSTRAINT {constraint_name}")
            logger.info("All constraints dropped successfully.")

            # Drop all indexes
            indexes = session.run("SHOW INDEXES")
            for index in indexes:
                index_name = index["name"]
                session.run(f"DROP INDEX {index_name}")
            logger.info("All indexes dropped successfully.")

    except Exception as e:
        logger.error(f"Error clearing database: {e}")


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

    # Add constraints and indices
    add_constraints_and_indices(connector)

    # Insert population nodes
    bulk_insert_population_nodes(connector, populations)

    # Insert neuron nodes with mtype labels in chunks
    chunk_size = 1000
    for i in range(0, len(nodes), chunk_size):
        chunk = nodes[i : i + chunk_size]
        bulk_insert_neuron_nodes(connector, chunk)

    # Insert BELONGS_TO relationships in chunks
    for i in range(0, len(belongs_to_relations), chunk_size):
        chunk = belongs_to_relations[i : i + chunk_size]
        bulk_insert_belongs_to_relationships(connector, chunk)

    # Insert edges in chunks
    for i in range(0, len(edges), chunk_size):
        chunk = edges[i : i + chunk_size]
        bulk_insert_edges(connector, chunk)

    # Optionally add labels based on mtype
    add_labels_based_on_mtype(connector, apply_labels)

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
