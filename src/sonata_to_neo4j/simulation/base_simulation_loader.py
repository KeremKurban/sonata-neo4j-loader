import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import libsonata
import pandas as pd
from bluepysnap import Circuit
from sonata_to_neo4j.utils import Neo4jConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationLoader:
    def __init__(
        self,
        circuit_config_path: str,
        data_dir: str,
        node_population_to_load: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ):
        self.circuit = Circuit(circuit_config_path)
        self.data_dir = Path(data_dir)
        self.connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
        self.node_population_name = node_population_to_load
        self.node_set = None
        self.node_set_ids = None

    def load_config(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            config = json.load(f)
        self.node_set = config["attrs"]["target"]
        self.node_set_ids = self.circuit.nodes[self.node_population_name].ids(self.node_set)
        logger.info(f"Node set: {self.node_set}")

    def load_spike_data(self, config_path: str) -> pd.DataFrame:
        with open(config_path, "r") as f:
            config = json.load(f)

        cell_frequencies = config["coords"]["cell_frequency"]["data"]
        signal_frequencies = config["coords"]["signal_frequency"]["data"]
        data_dirs = config["data"]

        spike_data = []

        for i, cell_freq in enumerate(cell_frequencies):
            for j, signal_freq in enumerate(signal_frequencies):
                dir_path = os.path.join(str(self.data_dir.parent), data_dirs[i][j])
                file_path = f"{dir_path}/out.dat"
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path, sep="\t", header=0, names=["spike_time", "neuron_id"])
                    data["neuron_id"] -= 1
                    data["cell_frequency"] = cell_freq
                    data["signal_frequency"] = signal_freq
                    spike_data.append(data)
                else:
                    logger.warning(f"No spike data files found in {dir_path}.")

        return pd.concat(spike_data, ignore_index=True)

    def filter_spiked_neurons(self, spike_data: pd.DataFrame) -> pd.DataFrame:
        return spike_data[["neuron_id"]].drop_duplicates()

    def extract_spiked_neurons(self, spiked_neurons_df: pd.DataFrame) -> List[Dict[str, Any]]:
        spiked_neurons = []
        for pop_name in self.circuit.nodes.population_names:
            node_population = self.circuit.nodes[pop_name]
            node_storage = libsonata.NodeStorage(node_population.h5_filepath)
            population = node_storage.open_population(pop_name)
            spiked_ids = spiked_neurons_df["neuron_id"].tolist()
            selection = libsonata.Selection(spiked_ids)
            attr_types = list(population.attribute_names)
            attributes = {attr: population.get_attribute(attr, selection) for attr in attr_types}
            df = pd.DataFrame(attributes)
            df["id"] = spiked_ids
            df["population_name"] = pop_name
            spiked_neurons.extend(df.to_dict("records"))
        return spiked_neurons

    def extract_edges_between_spiked_neurons(self, spiked_neurons_df: pd.DataFrame) -> List[Dict[str, Any]]:
        edges_df_list = []
        spiked_neurons_set = set(spiked_neurons_df["neuron_id"])

        for pop_name in self.circuit.edges.population_names:
            edge_population = self.circuit.edges[pop_name]
            edge_storage = libsonata.EdgeStorage(edge_population.h5_filepath)
            population = edge_storage.open_population(pop_name)
            edge_ids = list(range(population.size))
            selection = population.connecting_edges(self.node_set_ids, self.node_set_ids)
            attr_types = list(population.attribute_names)
            attributes = {attr: population.get_attribute(attr, selection) for attr in attr_types}
            source_node_ids = [population.source_node(eid) for eid in edge_ids]
            target_node_ids = [population.target_node(eid) for eid in edge_ids]

            df = pd.DataFrame(attributes)
            df["source_node_id"] = source_node_ids
            df["target_node_id"] = target_node_ids

            mask = df["source_node_id"].isin(spiked_neurons_set) & df["target_node_id"].isin(spiked_neurons_set)
            df = df[mask]

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

    def run(self) -> None:
        raise NotImplementedError("Subclasses should implement this method.")