import pandas as pd
import os

def load_spike_data(data_dir: str) -> pd.DataFrame:
    """
    Load spike data from out.dat files in the specified directory.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing out.dat files.

    Returns
    -------
    pd.DataFrame
        DataFrame containing spike times and neuron IDs.
    """
    spike_data = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file == "out.dat":
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path, sep='\t', header=None, names=['spike_time', 'neuron_id'])
                data['neuron_id'] -= 1  # Adjust for 1-indexed neuron IDs
                spike_data.append(data)
    return pd.concat(spike_data, ignore_index=True)

def bulk_insert_spike_nodes(connector: Neo4jConnector, spikes: List[Dict[str, Any]]) -> None:
    """
    Insert spike nodes into Neo4j in bulk.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    spikes : list of dict
        A list of dictionaries containing spike properties.
    """
    query = """
    UNWIND $spikes AS spike
    CREATE (s:Spike {time: spike.spike_time})
    """
    try:
        with connector.driver.session() as session:
            session.run(query, spikes=spikes)
            logger.info("Spike nodes inserted successfully.")
    except Exception as e:
        logger.error(f"Error inserting spike nodes: {e}")

def create_has_spike_relationships(connector: Neo4jConnector, spikes: List[Dict[str, Any]]) -> None:
    """
    Create HAS_SPIKE relationships between Neuron nodes and Spike nodes.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    spikes : list of dict
        A list of dictionaries containing spike properties.
    """
    query = """
    UNWIND $spikes AS spike
    MATCH (n:Neuron {id: spike.neuron_id})
    MATCH (s:Spike {time: spike.spike_time})
    MERGE (n)-[:HAS_SPIKE]->(s)
    """
    try:
        with connector.driver.session() as session:
            session.run(query, spikes=spikes)
            logger.info("HAS_SPIKE relationships created successfully.")
    except Exception as e:
        logger.error(f"Error creating HAS_SPIKE relationships: {e}")

def main(circuit_config_path: str, data_dir: str, apply_labels: bool = False) -> None:
    # ... existing code ...
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    node_proportion = float(os.getenv("NODE_PROPORTION", 1.0))
    edge_proportion = float(os.getenv("EDGE_PROPORTION", 1.0))
    node_set = os.getenv("NODE_SET")
    apply_mtype_labels = os.getenv("APPLY_MTYPE_LABELS")

    # Initialize BluePySnap Circuit
    circuit = Circuit(circuit_config_path)

    # Connect to Neo4j
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    # Load spike data
    # Insert neuron nodes with mtype labels in chunks
    chunk_size = 1000
    for i in range(0, len(nodes), chunk_size):
        chunk = nodes[i : i + chunk_size]
        bulk_insert_neuron_nodes(connector, chunk)

    spike_data_df = load_spike_data(data_dir)
    spikes = spike_data_df.to_dict("records")

    # Insert spike nodes in chunks
    chunk_size = 1000
    for i in range(0, len(spikes), chunk_size):
        chunk = spikes[i : i + chunk_size]
        bulk_insert_spike_nodes(connector, chunk)

    # Create HAS_SPIKE relationships
    create_has_spike_relationships(connector, spikes)


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
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing spike data.",
    )

    args = parser.parse_args()

    main(args.circuit_config, args.data_dir)