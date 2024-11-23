import unittest
from sonata_to_neo4j.circuit.circuit_loader import load_circuit
from unittest.mock import patch

class TestCircuitLoader(unittest.TestCase):

    @patch('src.circuit.circuit_loader.Neo4jDriver')
    def test_load_circuit(self, MockNeo4jDriver):
        # Arrange
        mock_driver = MockNeo4jDriver.return_value
        mock_session = mock_driver.session.return_value
        mock_session.run.return_value = None

        # Act
        result = load_circuit('path/to/circuit/file')

        # Assert
        mock_driver.session.assert_called_once()
        mock_session.run.assert_called()  # Check if the run method was called
        self.assertIsNone(result)  # Assuming the function returns None

if __name__ == '__main__':
    unittest.main()