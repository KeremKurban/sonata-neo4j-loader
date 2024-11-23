import argparse
import os

from circuit.circuit_loader import load_circuit
from dotenv import load_dotenv
from neo4j_connector import Neo4jConnector


def main(circuit_config_path: str, simulation_config: dict):
    # Load environment variables
    load_dotenv()

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    node_proportion = float(os.getenv("NODE_PROPORTION", 1.0))
    edge_proportion = float(os.getenv("EDGE_PROPORTION", 1.0))
    node_set = os.getenv("NODE_SET")

    # Connect to Neo4j
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)

    # Load circuit
    load_circuit(
        circuit_config_path, connector, node_proportion, edge_proportion, node_set
    )

    # Get simulation
    # simulation = SimulationTypeA(config=simulation_config)

    # Close connection
    connector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process SONATA files and run simulations."
    )
    parser.add_argument(
        "--circuit_config",
        type=str,
        required=True,
        help="Path to the circuit_config.JSON file.",
    )
    parser.add_argument(
        "--simulation_config",
        type=dict,
        required=False,
        help="Simulation config file, if exists.",
    )

    args = parser.parse_args()

    main(args.circuit_config, args.simulation_config)
