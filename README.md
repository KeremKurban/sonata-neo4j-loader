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

## Load Circuit

To load a SONATA circuit and simulation results into Neo4j, run the following command:

```bash
python src/main.py --circuit_config path/to/circuit_config.json --simulation_config path/to/simulation_config.json
```

This will create certain nodes and edge types as following:

### Nodes

- **Neuron**: 
  - Represents one node per neuron in the circuit target.

- **NeuronGroup**: 
  - Meta Node Class of population of Neurons.
  - **MType**: Neurons aggregated by their `mtype` property (e.g., SP_PC, L5_TTPC).
  - **SClass**: Neurons aggregated by their `synapse_class` property (e.g., EXC, INH).

- **Population**: 
  - Edge population name given in the SONATA file (e.g., hippocampus_neurons).

### Edges

- **SYNAPSE**: 
  - Represents synaptic connections between neurons, detailing the interaction and communication pathways.

- **BELONGS_TO_MTYPE**: 
  - Connects neurons to their respective MType groups, indicating the morphological type classification.

- **AGGREGATED_SYNAPSE**: 
  - Represents a collection of synapses aggregated for analysis or visualization purposes.

## Load Simulation
 
You can also give simulation config to load spiked neurons and their activity instead.

```bash
python src/main.py --simulation_config path/to/simulation_config.json
```
In addition to entities extracted from load circuit, this will also extract the following entities:

- **Spike** :
    Spike event described as node 

Edges:

- **HAS_SPIKE** :
    Represents the relationship between a neuron and its spike events, indicating which spikes are associated with which neurons.


## Node and Edge Entities Created 

### Nodes

- **Neuron Nodes**: Each neuron in the SONATA circuit is represented as a node in Neo4j. These nodes have properties such as `id` and `population_name`, and may include additional attributes extracted from the SONATA files.

- **NodeGroup Nodes**: These nodes are created based on specific properties of neurons, such as `mtype`. They help in organizing neurons into groups for easier querying and analysis.

- **SClass Nodes**: Represent synapse classes in the circuit. These nodes are linked to neurons based on the `synapse_class` property.

### Edges

- **SYNAPSE Edges**: These edges represent synaptic connections between neurons. Each edge has properties that describe the synapse, such as weight or type, and connects two neuron nodes.

- **BELONGS_TO_MTYPE Relationships**: These relationships connect neuron nodes to their respective NodeGroup nodes based on the `mtype` property.

- **BELONGS_TO_SCLASS Relationships**: These relationships link neurons to their corresponding SClass nodes, indicating the synapse class they belong to.

### Example Code References

- **Neuron Node Creation**: The logic for creating neuron nodes can be found in the `Neo4jConnector` class.
  ```python:sonata_to_neo4j/src/neo4j_connector.py
  ```

- **SYNAPSE Edge Creation**: The creation of synapse edges is handled in the `Neo4jConnector` class.
  ```python:sonata_to_neo4j/src/neo4j_connector.py
  ```

- **BELONGS_TO_SCLASS Relationships**: The creation of these relationships is detailed in the `neo4j_operations.py`.
  ```python:sonata_to_neo4j/src/circuit/neo4j_operations.py
  ```

This section provides a clear understanding of the entities being added to the Neo4j database and their relationships, helping users to better grasp the structure and purpose of the data model.


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.