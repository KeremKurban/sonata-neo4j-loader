import argparse
import os
import pandas as pd
from bluepysnap import Circuit
from dotenv import load_dotenv
from neo4j_connector import Neo4jConnector

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

def bulk_insert_population_nodes(connector, populations):
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

def main(circuit_config_path: str):
    """
    Main function to process SONATA files, extract nodes, edges, and populations, and insert them into Neo4j.
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

    # Extract populations and their nodes
    populations = [{'name': pop_name} for pop_name in circuit.nodes.population_names]
    nodes_df = []
    belongs_to_relations = []

    for pop_name in circuit.nodes.population_names:
        print('Extracting nodes from population:', pop_name)
        node_population = circuit.nodes[pop_name]
        node_df = node_population.get()
        node_df['population_name'] = pop_name
        nodes_df.append(node_df)
        belongs_to_relations.extend([
            {'node_id': node_id, 'population_name': pop_name}
            for node_id, row in node_df.iterrows()
        ])

    all_nodes_df = pd.concat(nodes_df, ignore_index=True)
    sampled_nodes_df = all_nodes_df.sample(frac=node_proportion)
    nodes = sampled_nodes_df.to_dict('records')
    breakpoint()

    # Extract and subsample edges
    edges_df = []
    for pop_name in circuit.edges.population_names:
        print('Extracting edges from population:', pop_name)
        edge_population = circuit.edges[pop_name]
        edge_df = edge_population.get() # FIXME: This is not working, and needs optimization while loading into neo4j
        edges_df.append(edge_df)

    all_edges_df = pd.concat(edges_df, ignore_index=True)
    sampled_edges_df = all_edges_df.sample(frac=edge_proportion)
    edges = sampled_edges_df.to_dict('records')

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
        bulk_insert_edges(connector, edges[i:i+chunk_size])

    # Close connection
    connector.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SONATA files and insert data into Neo4j.")
    parser.add_argument('--circuit_config', type=str, required=True, help='Path to the circuit configuration JSON file.')

    args = parser.parse_args()

    main(args.circuit_config)