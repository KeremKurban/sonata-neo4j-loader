import pytest
from unittest.mock import patch, MagicMock
from sonata_to_neo4j.circuit.circuit_loader import load_circuit

@pytest.mark.parametrize(
    "node_proportion, edge_proportion, node_set, circuit_config_path",
    [
        (1.0, 1.0, "Mosaic_A", "tests/test_data/circuit_sonata_quick_scx_multi_circuit/circuit_sonata.json"),
        (0.5, 0.5, "Mosaic_B", "tests/test_data/circuit_sonata_quick_scx_multi_circuit/circuit_sonata.json"),
        (0.0, 0.0, "Mosaic_A", "tests/test_data/circuit_sonata_quick_scx_multi_circuit/circuit_sonata.json"),
    ],
)
@patch("builtins.input", lambda _: "yes")
@patch("sonata_to_neo4j.utils.GraphDatabase.driver")
@patch("sonata_to_neo4j.circuit.circuit_loader.bulk_insert_neuron_nodes")
@patch("sonata_to_neo4j.circuit.circuit_loader.extract_nodes")
@patch("sonata_to_neo4j.circuit.circuit_loader.Circuit")
def test_load_circuit(
    mock_circuit,
    mock_extract_nodes,
    mock_bulk_insert_neuron_nodes,
    node_proportion,
    edge_proportion,
    node_set,
    circuit_config_path
):
    # Arrange
    mock_connector = MagicMock()
    mock_circuit_instance = MagicMock()
    mock_nodes = MagicMock()
    mock_circuit.return_value = mock_circuit_instance
    mock_extract_nodes.return_value = (mock_nodes, MagicMock(), MagicMock())

    # Act
    load_circuit(
        connector=mock_connector,
        node_proportion=node_proportion,
        edge_proportion=edge_proportion,
        node_set=node_set,
        circuit_config_path=circuit_config_path
    )

    # Assert
    mock_extract_nodes.assert_called_once_with(mock_circuit_instance, node_set, node_proportion)
    mock_bulk_insert_neuron_nodes.assert_called()