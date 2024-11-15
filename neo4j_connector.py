from neo4j import GraphDatabase

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
        tx.run("CREATE (n:Neuron {id: $id, properties: $properties})", id=node_id, properties=properties)

    def create_edge(self, start_node_id, end_node_id, properties):
        with self.driver.session() as session:
            session.execute_write(self._create_edge, start_node_id, end_node_id, properties)

    @staticmethod
    def _create_edge(tx, start_node_id, end_node_id, properties):
        tx.run("""
            MATCH (a:Neuron {id: $start_id}), (b:Neuron {id: $end_id})
            CREATE (a)-[:SYNAPSE {properties: $properties}]->(b)
        """, start_id=start_node_id, end_id=end_node_id, properties=properties)