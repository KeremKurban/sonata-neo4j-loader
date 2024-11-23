from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, node_id, properties):
        with self.driver.session() as session:
            session.execute_write(self._create_node, node_id, properties)

    @staticmethod
    def _create_node(tx, node_id, properties):
        tx.run(
            "CREATE (n:Neuron {id: $id, properties: $properties})",
            id=node_id,
            properties=properties,
        )

    def create_edge(self, start_node_id, end_node_id, properties):
        with self.driver.session() as session:
            session.execute_write(
                self._create_edge, start_node_id, end_node_id, properties
            )

    @staticmethod
    def _create_edge(tx, start_node_id, end_node_id, properties):
        tx.run(
            """
            MATCH (a:Neuron {id: $start_id}), (b:Neuron {id: $end_id})
            CREATE (a)-[:SYNAPSE {properties: $properties}]->(b)
            """,
            start_id=start_node_id,
            end_id=end_node_id,
            properties=properties,
        )

    def bulk_insert_neuron_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $nodes AS node
        MERGE (n:Neuron {id: node.id, population_name: node.population_name})
        SET n += node
        """
        try:
            with self.driver.session() as session:
                session.run(query, nodes=nodes)
                logger.info("Neuron nodes inserted successfully.")
        except Exception as e:
            logger.error(f"Error inserting neuron nodes: {e}")

    def bulk_insert_edges(self, edges: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $edges AS edge
        MATCH (a:Neuron {id: edge.source_node_id})
        MATCH (b:Neuron {id: edge.target_node_id})
        CREATE (a)-[r:SYNAPSE]->(b)
        SET r += edge
        """
        try:
            with self.driver.session() as session:
                session.run(query, edges=edges)
                logger.info("Edges inserted successfully.")
        except Exception as e:
            logger.error(f"Error inserting edges: {e}")

    def bulk_insert_spike_nodes(self, spikes: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $spikes AS spike
        CREATE (s:Spike {id: spike.id, spike_time: spike.spike_time})
        """
        try:
            with self.driver.session() as session:
                session.run(query, spikes=spikes)
                logger.info("Spike nodes inserted successfully.")
        except Exception as e:
            logger.error(f"Error inserting spike nodes: {e}")

    def create_has_spike_relationships(self, spikes: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $spikes AS spike
        MATCH (n:Neuron {id: spike.neuron_id})
        MATCH (s:Spike {id: spike.id})
        CREATE (n)-[:HAS_SPIKE]->(s)
        """
        try:
            with self.driver.session() as session:
                session.run(query, spikes=spikes)
                logger.info("HAS_SPIKE relationships created successfully.")
        except Exception as e:
            logger.error(f"Error creating HAS_SPIKE relationships: {e}")