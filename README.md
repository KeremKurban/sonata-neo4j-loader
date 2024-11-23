# SONATA to Neo4j Loader

## Overview

The `sonata-neo4j-loader` repository is designed to facilitate the loading of SONATA circuit data and simulation results into a Neo4j database. This tool is particularly useful for researchers and developers working with large-scale neural network models, allowing them to visualize and analyze the data within a graph database environment.

## Features

- **Circuit Loading**: Extracts and loads neuron and synapse data from SONATA circuit files into Neo4j.
- **Simulation Results Loading**: Processes and loads simulation results from CoreNeuron into Neo4j.
- **Modular Design**: Organized into separate modules for circuit and simulation handling, promoting scalability and maintainability.
- **Neo4j Integration**: Utilizes the Neo4j Python driver for efficient database operations.

## Directory Structure

- `src/`: Contains the main source code.
  - `circuit/`: Handles circuit data extraction and Neo4j operations.
    - `circuit_loader.py`: Main logic for loading circuit data.
    - `data_extraction.py`: Functions for extracting nodes and edges.
    - `neo4j_operations.py`: Functions for interacting with Neo4j.
  - `simulation/`: Handles simulation result processing.
    - `simulation_loader.py`: Base class for loading simulation results.
    - `sim_type_A.py`: Example of a specific simulation result handler.
  - `utils.py`: Utility functions shared across modules.
- `tests/`: Contains unit tests for the codebase.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sonata-neo4j-loader.git
   cd sonata-neo4j-loader
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   NODE_PROPORTION=1.0
   EDGE_PROPORTION=1.0
   NODE_SET=your_node_set
   ```

## Usage

To load a SONATA circuit and simulation results into Neo4j, run the following command:

python src/main.py --circuit_config path/to/circuit_config.json --simulation_config path/to/simulation_config.json

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.