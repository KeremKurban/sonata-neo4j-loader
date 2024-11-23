# import unittest
# from unittest.mock import patch

# from sonata_to_neo4j.simulation.simulation_loader import load_simulation_results


# class TestSimulationLoader(unittest.TestCase):
#     @patch("src.simulation.simulation_loader.Neo4jDriver")
#     def test_load_simulation_results(self, MockNeo4jDriver):
#         # Arrange
#         mock_driver = MockNeo4jDriver.return_value
#         mock_session = mock_driver.session.return_value
#         mock_session.run.return_value = None

#         # Act
#         result = load_simulation_results("path/to/simulation/results")

#         # Assert
#         mock_driver.session.assert_called_once()
#         mock_session.run.assert_called()  # Check if the run method was called
#         self.assertIsNone(result)  # Assuming the function returns None


# if __name__ == "__main__":
#     unittest.main()
