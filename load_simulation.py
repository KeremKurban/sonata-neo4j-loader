import os
import pandas as pd
from typing import Any, Dict, List, Tuple
from bluepysnap import Circuit
from neo4j_connector import Neo4jConnector
import libsonata
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Simulation:
    def __init__(self, circuit_config_path: str, data_dir: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.circuit = Circuit(circuit_config_path)
        self.data_dir = Path(data_dir)
        self.connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)

    def load_spike_data(self, config_path: str) -> pd.DataFrame:
        """
        Load spike data from out.dat files based on the configuration file.

        Parameters
        ----------
        config_path : str
            Path to the configuration JSON file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing spike times, neuron IDs, cell frequency, and signal frequency.
        """
        # Load the configuration file
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Extract frequencies and data directories
        cell_frequencies = config['coords']['cell_frequency']['data']
        signal_frequencies = config['coords']['signal_frequency']['data']
        data_dirs = config['data']

        spike_data = []

        # Iterate over the combinations of cell and signal frequencies
        for i, cell_freq in enumerate(cell_frequencies):
            for j, signal_freq in enumerate(signal_frequencies):
                # Get the corresponding directory for this combination
                dir_path = os.path.join(str(self.data_dir.parent), data_dirs[i][j])
                file_path = f'{dir_path}/out.dat'
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path, sep='\t', header=0, names=['spike_time', 'neuron_id'])
                    data['neuron_id'] -= 1  # Adjust for 1-indexed neuron IDs
                    data['cell_frequency'] = cell_freq
                    data['signal_frequency'] = signal_freq
                    spike_data.append(data)
                else:
                    logger.warning(f"No spike data files found in {dir_path}.")

        return pd.concat(spike_data, ignore_index=True)

    def filter_spiked_neurons(self, spike_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter neurons that have spiked.

        Parameters
        ----------
        spike_data : pd.DataFrame
            DataFrame containing spike times and neuron IDs.

        Returns
        -------
        pd.DataFrame
            DataFrame containing unique spiked neuron IDs.
        """
        return spike_data[['neuron_id']].drop_duplicates()

    def extract_spiked_neurons(self, spiked_neurons_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract spiked neurons from the circuit.

        Parameters
        ----------
        spiked_neurons_df : pd.DataFrame
            DataFrame containing unique spiked neuron IDs.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing spiked neuron properties.
        """
        spiked_neurons = []
        for pop_name in self.circuit.nodes.population_names:
            node_population = self.circuit.nodes[pop_name]
            node_storage = libsonata.NodeStorage(node_population.h5_filepath)
            population = node_storage.open_population(pop_name)
            spiked_ids = spiked_neurons_df['neuron_id'].tolist()
            selection = libsonata.Selection(spiked_ids)
            attr_types = list(population.attribute_names)
            attributes = {attr: population.get_attribute(attr, selection) for attr in attr_types}
            df = pd.DataFrame(attributes)
            df["id"] = spiked_ids
            df["population_name"] = pop_name
            spiked_neurons.extend(df.to_dict("records"))
        return spiked_neurons

    def extract_edges_between_spiked_neurons(self, spiked_neurons_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract edges between spiked neurons.

        Parameters
        ----------
        spiked_neurons_df : pd.DataFrame
            DataFrame containing unique spiked neuron IDs.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing edge properties.
        """
        edges_df_list = []
        spiked_neurons_set = set(spiked_neurons_df['neuron_id'])

        for pop_name in self.circuit.edges.population_names:
            edge_population = self.circuit.edges[pop_name]
            edge_storage = libsonata.EdgeStorage(edge_population.h5_filepath)
            population = edge_storage.open_population(pop_name)
            edge_ids = list(range(population.size))
            selection = libsonata.Selection(edge_ids)
            attr_types = list(population.attribute_names)
            attributes = {attr: population.get_attribute(attr, selection) for attr in attr_types}
            source_node_ids = [population.source_node(eid) for eid in edge_ids]
            target_node_ids = [population.target_node(eid) for eid in edge_ids]

            df = pd.DataFrame(attributes)
            df["source_node_id"] = source_node_ids
            df["target_node_id"] = target_node_ids
            breakpoint()
            # Filter edges where both source and target nodes are spiked neurons
            mask = df["source_node_id"].isin(spiked_neurons_set) & df["target_node_id"].isin(spiked_neurons_set)
            df = df[mask]

            # Remove duplicates based on source and target node IDs
            df = df.drop_duplicates(subset=["source_node_id", "target_node_id"])
            edges_df_list.append(df)

        if edges_df_list:
            all_edges_df = pd.concat(edges_df_list, ignore_index=True)
            edges = all_edges_df.to_dict("records")
            logger.info(f"Total unique edges found: {len(edges)}")
        else:
            edges = []
            logger.info("No edges found between spiked neurons.")
        
        return edges

    def insert_spiked_neurons_and_edges(self, spiked_neurons: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        """
        Insert spiked neurons and edges into Neo4j.

        Parameters
        ----------
        spiked_neurons : List[Dict[str, Any]]
            List of dictionaries containing spiked neuron properties.
        edges : List[Dict[str, Any]]
            List of dictionaries containing edge properties.
        """
        # Insert spiked neurons
        chunk_size = 1000
        for i in range(0, len(spiked_neurons), chunk_size):
            chunk = spiked_neurons[i : i + chunk_size]
            self.bulk_insert_neuron_nodes(chunk)

        # Insert edges
        for i in range(0, len(edges), chunk_size):
            chunk = edges[i : i + chunk_size]
            self.bulk_insert_edges(chunk)

    def bulk_insert_neuron_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        """
        Insert neuron nodes into Neo4j in bulk.

        Parameters
        ----------
        nodes : list of dict
            A list of dictionaries containing neuron properties.
        """
        query = """
        UNWIND $nodes AS node
        MERGE (n:Neuron {id: node.id, population_name: node.population_name})
        SET n += node
        """
        try:
            with self.connector.driver.session() as session:
                session.run(query, nodes=nodes)
                logger.info("Neuron nodes inserted successfully.")
        except Exception as e:
            logger.error(f"Error inserting neuron nodes: {e}")

    def bulk_insert_edges(self, edges: List[Dict[str, Any]]) -> None:
        """
        Insert edges into Neo4j in bulk using UNWIND.

        Parameters
        ----------
        edges : list of dict
            A list of dictionaries containing edge properties.
        """
        query = """
        UNWIND $edges AS edge
        MATCH (a:Neuron {id: edge.source_node_id})
        MATCH (b:Neuron {id: edge.target_node_id})
        CREATE (a)-[r:SYNAPSE]->(b)
        SET r += edge
        """
        try:
            with self.connector.driver.session() as session:
                session.run(query, edges=edges)
                logger.info("Edges inserted successfully.")
        except Exception as e:
            logger.error(f"Error inserting edges: {e}")

    def run(self) -> None:
        """
        Run the simulation to load spiked neurons and edges into Neo4j.
        """
        # Load spike data
        spike_data_df = self.load_spike_data(f"{str(self.data_dir)}/config.json")
        # Filter spiked neurons
        spiked_neurons_df = self.filter_spiked_neurons(spike_data_df)

        # Extract spiked neurons and edges
        spiked_neurons = self.extract_spiked_neurons(spiked_neurons_df)
        edges = self.extract_edges_between_spiked_neurons(spiked_neurons_df)

        # Insert spiked neurons and edges into Neo4j
        self.insert_spiked_neurons_and_edges(spiked_neurons, edges)

        # Close connection
        self.connector.close()

if __name__ == "__main__":
    # Example usage
    simulation = Simulation(
        circuit_config_path="/Users/kurban/Documents/bbp/neo4j_sonata/20211110-BioM_slice10/sonata/circuit_config_local.json",
        data_dir="/Users/kurban/Documents/bbp/neo4j_sonata/examples/simulations/CA1.20211110-BioM/22cc40cf-d0c2-4c0c-8bca-e9a4a96c16bb",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )
    simulation.run()