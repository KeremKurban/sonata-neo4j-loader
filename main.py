import argparse
import os
import pandas as pd
from bluepysnap import Circuit
from dotenv import load_dotenv
from neo4j_connector import Neo4jConnector
import libsonata


def add_constraints_and_indices(connector):
    """
    Add constraints and indices to the Neo4j database for efficient imports and querying.
    """
    with connector.driver.session() as session:
        # Create constraints for Neurons
        session.run("CREATE CONSTRAINT FOR (n:Neuron) REQUIRE n.id IS UNIQUE;")
        session.run("CREATE INDEX FOR (n:Neuron) ON (n.id);")

        # Create constraints for Populations
        session.run("CREATE CONSTRAINT FOR (p:Population) REQUIRE p.name IS UNIQUE;")
        session.run("CREATE INDEX FOR (p:Population) ON (p.name);")


def bulk_insert_population_nodes(connector, populations: list):
    """
    Insert population nodes into Neo4j in bulk.
    """
    with connector.driver.session() as session:
        query = """
        UNWIND $populations AS population
        MERGE (p:Population {name: population.name})
        SET p += population
        """
        session.run(query, populations=populations)


def bulk_insert_neuron_nodes(connector, nodes):
    """
    Insert neuron nodes into Neo4j in bulk, assigning their mtype property as a label.
    """
    with connector.driver.session() as session:
        query = """
        UNWIND $nodes AS node
        CALL apoc.create.node(['Neuron', node.mtype], node) YIELD node AS n
        RETURN n
        """
        session.run(query, nodes=nodes)


def bulk_insert_belongs_to_relationships(connector, belongs_to_relations):
    """
    Create BELONGS_TO relationships between neurons and populations in bulk.
    """
    with connector.driver.session() as session:
        query = """
        UNWIND $relations AS rel
        MATCH (n:Neuron {id: rel.node_id}), (p:Population {name: rel.population_name})
        MERGE (n)-[:BELONGS_TO]->(p)
        """
        session.run(query, relations=belongs_to_relations)


def bulk_insert_edges(connector, edges):
    """
    Insert edges into Neo4j in bulk using UNWIND.
    """
    with connector.driver.session() as session:
        query = """
        UNWIND $edges AS edge
        MATCH (a:Neuron {id: edge.source_node_id}), (b:Neuron {id: edge.target_node_id})
        MERGE (a)-[r:SYNAPSE]->(b)
        SET r += edge
        """
        session.run(query, edges=edges)


def extract_nodes(circuit, proportion):
    """
    Extract and subsample nodes from node populations using libsonata and bluepy Circuit entity
    """
    nodes_df = []
    belongs_to_relations = []
    populations = []
    for pop_name in circuit.nodes.population_names:
        print("Extracting nodes from population:", pop_name)
        node_population = circuit.nodes[pop_name]
        node_storage = libsonata.NodeStorage(node_population.h5_filepath)
        population = node_storage.open_population(pop_name)
        node_ids = range(population.size)
        selection = libsonata.Selection(range(population.size))
        attr_types = population.attribute_names
        attr_types = list(attr_types)
        attributes = {
            attr: population.get_attribute(attr, selection) for attr in attr_types
        }
        df = pd.DataFrame(attributes)
        df["id"] = node_ids
        df["population_name"] = pop_name
        nodes_df.append(df)
        belongs_to_relations.extend(
            [{"node_id": node_id, "population_name": pop_name} for node_id in node_ids]
        )
        populations.append(
            {
                "name": pop_name,
                "size": population.size,
                "population_name": pop_name,
                "attr_types": attr_types,
            }
        )
    all_nodes_df = pd.concat(nodes_df, ignore_index=True)
    sampled_nodes_df = all_nodes_df.sample(frac=proportion)
    nodes = sampled_nodes_df.to_dict("records")
    return nodes, belongs_to_relations, populations


def extract_edges(circuit, proportion):
    """
    Extract and subsample edges from edge populations using libsonata and bluepy Circuit entity
    """
    edges_df = []
    for pop_name in circuit.edges.population_names:
        print("Extracting edges from population:", pop_name)
        edge_population = circuit.edges[pop_name]
        edge_storage = libsonata.EdgeStorage(edge_population.h5_filepath)
        population = edge_storage.open_population(pop_name)
        edge_ids = range(population.size)
        selection = libsonata.Selection(range(population.size))
        attr_types = population.attribute_names
        attr_types = list(attr_types)
        attributes = {
            attr: population.get_attribute(attr, selection) for attr in attr_types
        }
        source_nodes = [population.source_node(eid) for eid in edge_ids]
        target_nodes = [population.target_node(eid) for eid in edge_ids]
        df = pd.DataFrame(attributes)
        df["source_node_id"] = source_nodes
        df["target_node_id"] = target_nodes
        edges_df.append(df)
    all_edges_df = pd.concat(edges_df, ignore_index=True)
    sampled_edges_df = all_edges_df.sample(frac=proportion)
    edges = sampled_edges_df.to_dict("records")
    return edges


def main(circuit_config_path: str):
    """
    Main function to process SONATA files, extract nodes, edges, and populations, and insert them into Neo4j.
    """
    # Load environment variables
    load_dotenv()

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    node_proportion = float(os.getenv("NODE_PROPORTION", 1.0))
    edge_proportion = float(os.getenv("EDGE_PROPORTION", 1.0))

    # Initialize BluePySnap Circuit
    circuit = Circuit(circuit_config_path)

    # Extract nodes and edges using functions above
    nodes, belongs_to_relations, populations = extract_nodes(circuit, node_proportion)
    edges = extract_edges(circuit, edge_proportion)

    breakpoint()
    # Connect to Neo4j
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)

    # Add constraints and indices
    add_constraints_and_indices(connector)

    # Insert population nodes
    bulk_insert_population_nodes(connector, populations)

    # Insert neuron nodes with mtype labels
    bulk_insert_neuron_nodes(connector, nodes)

    # Insert BELONGS_TO relationships
    bulk_insert_belongs_to_relationships(connector, belongs_to_relations)

    # Insert edges in bulk
    chunk_size = 1000
    for i in range(0, len(edges), chunk_size):
        bulk_insert_edges(connector, edges[i : i + chunk_size])

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
