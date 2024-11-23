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
@patch("neo4j.GraphDatabase.driver")
def test_load_circuit(MockNeo4jDriver, node_proportion, edge_proportion, node_set, circuit_config_path):
    # Arrange
    mock_driver = MockNeo4jDriver.return_value
    mock_session = mock_driver.session.return_value
    mock_session.run.return_value = None

    # Create a mock connector
    mock_connector = MagicMock()

    # Act
    result = load_circuit(
        connector=mock_connector,
        node_proportion=node_proportion,
        edge_proportion=edge_proportion,
        node_set=node_set,
        circuit_config_path=circuit_config_path
    )

    # Assert
    mock_driver.session.assert_called_once()
    mock_session.run.assert_called()  # Check if the run method was called
    assert result is None  # Assuming the function returns None