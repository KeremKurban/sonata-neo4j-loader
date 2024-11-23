import pytest
from unittest.mock import patch, MagicMock
from sonata_to_neo4j.circuit.circuit_loader import load_circuit

class TestCircuitLoader:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.mock_connector = MagicMock()
        self.mock_circuit_instance = MagicMock()
        self.mock_nodes = MagicMock()
        self.mock_edges = MagicMock()

    @pytest.mark.parametrize(
        "node_proportion, edge_proportion, node_set, circuit_config_path",
        [
            (1.0, 1.0, "Mosaic_A", "tests/test_data/circuit_sonata_quick_scx_multi_circuit/circuit_sonata.json"),
            (0.5, 0.5, "Mosaic_B", "tests/test_data/circuit_sonata_quick_scx_multi_circuit/circuit_sonata.json"),
            (0.0, 0.0, "Mosaic_A", "tests/test_data/circuit_sonata_quick_scx_multi_circuit/circuit_sonata.json"),
        ],
    )
    @patch("sonata_to_neo4j.circuit.circuit_loader.extract_nodes")
    @patch("sonata_to_neo4j.circuit.circuit_loader.Circuit")
    def test_extract_nodes_called_correctly(
        self,
        mock_circuit,
        mock_extract_nodes,
        node_proportion,
        edge_proportion,
        node_set,
        circuit_config_path
    ):
        # Arrange
        mock_circuit.return_value = self.mock_circuit_instance
        mock_extract_nodes.return_value = (self.mock_nodes, MagicMock(), MagicMock())

        # Act
        load_circuit(
            connector=self.mock_connector,
            node_proportion=node_proportion,
            edge_proportion=edge_proportion,
            node_set=node_set,
            circuit_config_path=circuit_config_path
        )

        # Assert
        mock_extract_nodes.assert_called_once_with(self.mock_circuit_instance, node_set, node_proportion)

    @patch("sonata_to_neo4j.circuit.circuit_loader.extract_nodes")
    @patch("sonata_to_neo4j.circuit.circuit_loader.Circuit")
    def test_circuit_instance_created(self, mock_circuit, mock_extract_nodes):
        # Arrange
        mock_circuit.return_value = self.mock_circuit_instance
        mock_extract_nodes.return_value = (self.mock_nodes, MagicMock(), MagicMock())

        # Act
        load_circuit(
            connector=self.mock_connector,
            node_proportion=1.0,
            edge_proportion=1.0,
            node_set="Mosaic_A",
            circuit_config_path="tests/test_data/circuit_sonata_quick_scx_multi_circuit/circuit_sonata.json"
        )

        # Assert
        mock_circuit.assert_called_once()