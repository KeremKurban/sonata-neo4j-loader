import argparse
import logging
import os
from dotenv import load_dotenv
from sonata_to_neo4j.simulation.base_simulation_loader import SimulationLoader
from sonata_to_neo4j.circuit.circuit_loader import load_circuit,clear_database
from sonata_to_neo4j.utils import Neo4jConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class SimulationCampaign(SimulationLoader):
    def load_results(self):
        self.load_config(f"{str(self.data_dir)}/config.json")
        spike_data_df = self.load_spike_data(f"{str(self.data_dir)}/config.json")
        spiked_neurons_df = self.filter_spiked_neurons(spike_data_df)
        return spike_data_df, spiked_neurons_df

    def process_results(self, spike_data_df, spiked_neurons_df):
        spiked_neurons = self.extract_spiked_neurons(spiked_neurons_df)
        edges = self.extract_edges_between_spiked_neurons(spiked_neurons_df)
        spikes = self.extract_spikes(spike_data_df)
        self.insert_spiked_neurons_edges_and_spikes(spiked_neurons, edges, spikes)

    def extract_spikes(self, spike_data_df):
        spikes = []
        for _, row in spike_data_df.iterrows():
            spike_data = {
                "id": f"spike_{row['neuron_id']}_{row['spike_time']}",
                "spike_time": row['spike_time'],
                "neuron_id": row['neuron_id']
            }
            spikes.append(spike_data)
        return spikes

    def insert_spiked_neurons_edges_and_spikes(self, spiked_neurons, edges, spikes):
        self.connector.bulk_insert_neuron_nodes(spiked_neurons)
        self.connector.bulk_insert_edges(edges)
        self.connector.bulk_insert_spike_nodes(spikes)
        self.connector.create_has_spike_relationships(spikes)

    def run(self):
        spike_data_df, spiked_neurons_df = self.load_results()
        self.process_results(spike_data_df, spiked_neurons_df)

def main():
    parser = argparse.ArgumentParser(description="Load SONATA simulation data into Neo4j.")
    parser.add_argument('--circuit_config_path', type=str, required=True, help='Path to the circuit configuration JSON file.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing simulation data.')
    args = parser.parse_args()

    # Retrieve environment variables
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    node_population = os.getenv("NODE_POPULATION")
    node_proportion = float(os.getenv("NODE_PROPORTION", 1))
    edge_proportion = float(os.getenv("EDGE_PROPORTION", 0.001))

    # Initialize Neo4jConnector
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)

    clear_database(connector)
    
    # Load circuit data
    load_circuit(
        circuit_config_path=args.circuit_config_path,
        connector=connector,
        node_proportion=node_proportion,
        edge_proportion=edge_proportion,
        node_set=node_population
    )

    # Run simulation
    simulation = BasicSimulation(
        circuit_config_path=args.circuit_config_path,
        data_dir=args.data_dir,
        node_population_to_load=node_population,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )
    simulation.run()

if __name__ == "__main__":
    main()