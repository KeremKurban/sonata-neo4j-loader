from neo4j_connector import Neo4jConnector
import logging
from typing import Any, Dict, List
import functools

logger = logging.getLogger(__name__)

def log_entity_count(entity_type: str, label_or_type: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(connector: Neo4jConnector, *args, **kwargs):
            # Define a query to count the entities
            if entity_type == "node":
                count_query = f"MATCH (n:{label_or_type}) RETURN count(n) AS entity_count"
            elif entity_type == "relationship":
                count_query = f"MATCH ()-[r:{label_or_type}]->() RETURN count(r) AS entity_count"
            else:
                logger.error(f"Unsupported entity type: {entity_type}")
                return
            
            # Count entities before execution
            try:
                with connector.driver.session() as session:
                    result = session.run(count_query)
                    count_before = result.single()["entity_count"]
            except Exception as e:
                logger.error(f"Error counting {label_or_type} {entity_type}s before execution: {e}")
                return
            
            # Execute the original function
            func(connector, *args, **kwargs)
            
            # Count entities after execution
            try:
                with connector.driver.session() as session:
                    result = session.run(count_query)
                    count_after = result.single()["entity_count"]
                    created_count = count_after - count_before
                    logger.info(f"Total '{label_or_type}' {entity_type}s created: {created_count}.")
            except Exception as e:
                logger.error(f"Error counting {label_or_type} {entity_type}s after execution: {e}")
        
        return wrapper
    return decorator

def clear_database(connector: Neo4jConnector) -> None:
    """
    Clear all nodes, relationships, constraints, and indexes in the Neo4j database using APOC's truncate procedure.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    """
    try:
        with connector.driver.session() as session:
            # Truncate the database
            session.run("CALL apoc.periodic.truncate()")
            logger.info("Database truncated successfully.")

    except Exception as e:
        logger.error(f"Error clearing database: {e}")



def create_sclass_nodes(connector: Neo4jConnector, nodes: List[Dict[str, Any]]) -> None:
    """
    Create SClass nodes in Neo4j based on the 'synapse_class' property.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    nodes : list of dict
        A list of dictionaries containing node properties.
    """
    unique_synapse_classes = {
        node["synapse_class"]
        for node in nodes
        if "synapse_class" in node and node["synapse_class"] is not None
    }
    query = """
    UNWIND $values AS value
    MERGE (s:SClass {name: value})
    """
    try:
        with connector.driver.session() as session:
            session.run(query, values=list(unique_synapse_classes))
            logger.info("SClass nodes created successfully.")
    except Exception as e:
        logger.error(f"Error creating SClass nodes: {e}")


def create_neuron_belongs_to_sclass_relationships(
    connector: Neo4jConnector, nodes: List[Dict[str, Any]]
) -> None:
    """
    Create BELONGS_TO_SCLASS relationships between neurons and SClass nodes.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    nodes : list of dict
        A list of dictionaries containing node properties.
    """
    query = """
    UNWIND $nodes AS node
    MATCH (n:Neuron {id: node.id, population_name: node.population_name})
    MATCH (s:SClass {name: node.synapse_class})
    MERGE (n)-[:BELONGS_TO_SCLASS]->(s)
    """
    try:
        with connector.driver.session() as session:
            session.run(query, nodes=nodes)
            logger.info("BELONGS_TO_SCLASS relationships created successfully.")
    except Exception as e:
        logger.error(f"Error creating BELONGS_TO_SCLASS relationships: {e}")


def create_sclass_relationships(connector: Neo4jConnector) -> None:
    """
    Create relationships between SClass nodes based on connections between Neuron nodes.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    """
    query = """
    MATCH (n1:Neuron)-[r:SYNAPSE]->(n2:Neuron)
    MATCH (s1:SClass {name: n1.synapse_class}), (s2:SClass {name: n2.synapse_class})
    WITH s1, s2, avg(r.conductance) AS avg_conductance, avg(r.delay) AS avg_delay
    MERGE (s1)-[rs:AGGREGATED_SYNAPSE]->(s2)
    SET rs.avg_conductance = avg_conductance,
        rs.avg_delay = avg_delay
    """
    try:
        with connector.driver.session() as session:
            session.run(query)
            logger.info("SClass relationships created successfully.")
    except Exception as e:
        logger.error(f"Error creating SClass relationships: {e}")


def create_nodegroup_nodes(
    connector: Neo4jConnector,
    property_name: str,
    label: str,
    nodes: List[Dict[str, Any]],
) -> None:
    """
    Create NodeGroup nodes in Neo4j based on a specified property and label.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    property_name : str
        The property name to base the NodeGroup nodes on (e.g., 'mtype').
    label : str
        The label to assign to the NodeGroup nodes (e.g., 'MType').
    nodes : list of dict
        A list of dictionaries containing node properties.
    """
    unique_values = {
        node[property_name]
        for node in nodes
        if property_name in node and node[property_name] is not None
    }
    query = f"""
    UNWIND $values AS value
    MERGE (n:NodeGroup:{label} {{name: value}})
    """
    try:
        with connector.driver.session() as session:
            session.run(query, values=list(unique_values))
            logger.info(f"NodeGroup {label} nodes created successfully.")
    except Exception as e:
        logger.error(f"Error creating NodeGroup {label} nodes: {e}")


def create_neuron_belongs_to_nodegroup_relationships(
    connector: Neo4jConnector,
    property_name: str,
    label: str,
    nodes: List[Dict[str, Any]],
) -> None:
    """
    Create BELONGS_TO relationships between nodes and NodeGroup nodes based on a specified property.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    property_name : str
        The property name to base the relationships on (e.g., 'mtype').
    label : str
        The label of the NodeGroup nodes (e.g., 'MType').
    nodes : list of dict
        A list of dictionaries containing node properties.
    """
    query = f"""
    UNWIND $nodes AS node
    MATCH (n:Neuron {{id: node.id, population_name: node.population_name}})
    MATCH (g:NodeGroup:{label} {{name: node.{property_name}}})
    MERGE (n)-[:BELONGS_TO_{label.upper()}]->(g)
    """
    try:
        with connector.driver.session() as session:
            session.run(query, nodes=nodes)
            logger.info(
                f"BELONGS_TO_{label.upper()} relationships created successfully."
            )
    except Exception as e:
        logger.error(f"Error creating BELONGS_TO_{label.upper()} relationships: {e}")


@log_entity_count(entity_type="relationship", label_or_type="AGGREGATED_SYNAPSE")
def create_nodegroup_relationships(connector: Neo4jConnector, property_name: str) -> None:
    """
    Create relationships between NodeGroup nodes based on connections between Neuron nodes.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    property_name : str
        The property name to base the NodeGroup relationships on (e.g., 'mtype').
    """
    query = f"""
    MATCH (n1:Neuron)-[r:SYNAPSE]->(n2:Neuron)
    MATCH (g1:NodeGroup {{name: n1.{property_name}}}), (g2:NodeGroup {{name: n2.{property_name}}})
    WITH g1, g2, avg(r.conductance) AS avg_conductance, avg(r.delay) AS avg_delay
    MERGE (g1)-[rg:AGGREGATED_SYNAPSE]->(g2)
    SET rg.avg_conductance = avg_conductance,
        rg.avg_delay = avg_delay
    """
    try:
        with connector.driver.session() as session:
            session.run(query)
            logger.info(f"NodeGroup relationships based on '{property_name}' property created successfully.")
    except Exception as e:
        logger.error(f"Error creating NodeGroup relationships: {e}")

# def create_nodegroup_relationships(connector: Neo4jConnector, property_name: str) -> None:
#     """
#     Create relationships between NodeGroup nodes based on connections between Neuron nodes.

#     Parameters
#     ----------
#     connector : Neo4jConnector
#         An instance of the Neo4jConnector class.
#     property_name : str
#         The property name to base the NodeGroup relationships on (e.g., 'mtype').
#     """
#     query = f"""
#     MATCH (n1:Neuron)-[r:SYNAPSE]->(n2:Neuron)
#     MATCH (g1:NodeGroup {{name: n1.{property_name}}}), (g2:NodeGroup {{name: n2.{property_name}}})
#     WITH g1, g2, avg(r.conductance) AS avg_conductance, avg(r.delay) AS avg_delay
#     MERGE (g1)-[rg:AGGREGATED_SYNAPSE]->(g2)
#     SET rg.avg_conductance = avg_conductance,
#         rg.avg_delay = avg_delay
#     RETURN count(rg) AS relationship_count
#     """
#     try:
#         with connector.driver.session() as session:
#             result = session.run(query)
#             relationship_count = result.single()["relationship_count"]
#             logger.info(f"NodeGroup relationships based on '{property_name}' property created successfully. Total relationships created: {relationship_count}.")
#     except Exception as e:
#         logger.error(f"Error creating NodeGroup relationships: {e}")


def add_labels_based_on_mtype(connector: Neo4jConnector, apply_labels: bool) -> None:
    """
    Optionally add labels to Neuron nodes based on their 'mtype' property.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    apply_labels : bool
        Whether to apply labels based on the 'mtype' property.
    """
    if not apply_labels:
        return

    query = """
    MATCH (n:Neuron)
    WHERE n.mtype IS NOT NULL
    CALL apoc.create.addLabels(n, [n.mtype]) YIELD node
    RETURN node.mtype AS mtype, labels(node) AS labels
    LIMIT 25
    """
    try:
        with connector.driver.session() as session:
            result = session.run(query)
            for record in result:
                logger.info(
                    f"Added label {record['mtype']} to node with labels {record['labels']}"
                )
    except Exception as e:
        logger.error(f"Error adding labels based on mtype: {e}")


def bulk_insert_population_nodes(
    connector: Neo4jConnector, populations: List[Dict[str, Any]]
) -> None:
    """
    Insert population nodes into Neo4j in bulk.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    populations : list of dict
        A list of dictionaries containing population properties.
    """
    query = """
    UNWIND $populations AS population
    MERGE (p:Population {name: population.name})
    SET p += population
    """
    try:
        with connector.driver.session() as session:
            session.run(query, populations=populations)
            logger.info("Population nodes inserted successfully.")
    except Exception as e:
        logger.error(f"Error inserting population nodes: {e}")


def bulk_insert_neuron_nodes(
    connector: Neo4jConnector, nodes: List[Dict[str, Any]]
) -> None:
    """
    Insert neuron nodes into Neo4j in bulk.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    nodes : list of dict
        A list of dictionaries containing neuron properties.
    """
    query = """
    UNWIND $nodes AS node
    MERGE (n:Neuron {id: node.id, population_name: node.population_name})
    SET n += node
    """
    try:
        with connector.driver.session() as session:
            session.run(query, nodes=nodes)
            logger.info("Neuron nodes inserted successfully.")
    except Exception as e:
        logger.error(f"Error inserting neuron nodes: {e}")


def bulk_insert_belongs_to_relationships(
    connector: Neo4jConnector, belongs_to_relations: List[Dict[str, Any]]
) -> None:
    """
    Create BELONGS_TO relationships between neurons and populations in bulk.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    belongs_to_relations : list of dict
        A list of dictionaries with 'node_id' and 'population_name' keys.
    """
    query = """
    UNWIND $relations AS rel
    MATCH (n:Neuron {id: rel.node_id, population_name: rel.population_name})
    MATCH (p:Population {name: rel.population_name})
    MERGE (n)-[:BELONGS_TO]->(p)
    """
    try:
        with connector.driver.session() as session:
            session.run(query, relations=belongs_to_relations)
            logger.info("BELONGS_TO relationships created successfully.")
    except Exception as e:
        logger.error(f"Error creating BELONGS_TO relationships: {e}")


def bulk_insert_edges(connector: Neo4jConnector, edges: List[Dict[str, Any]]) -> None:
    """
    Insert edges into Neo4j in bulk using UNWIND.

    Parameters
    ----------
    connector : Neo4jConnector
        An instance of the Neo4jConnector class.
    edges : list of dict
        A list of dictionaries containing edge properties.
    """
    query = """
    UNWIND $edges AS edge
    MATCH (a:Neuron {id: edge.source_node_id, population_name: edge.source_population_name})
    MATCH (b:Neuron {id: edge.target_node_id, population_name: edge.target_population_name})
    CREATE (a)-[r:SYNAPSE]->(b)
    SET r += edge.properties
    """
    # Prepare edges by separating properties
    transformed_edges = [
        {
            "source_node_id": edge["source_node_id"],
            "source_population_name": edge["source_population_name"],
            "target_node_id": edge["target_node_id"],
            "target_population_name": edge["target_population_name"],
            "properties": {
                k: v
                for k, v in edge.items()
                if k
                not in [
                    "source_node_id",
                    "source_population_name",
                    "target_node_id",
                    "target_population_name",
                ]
            },
        }
        for edge in edges
    ]
    try:
        with connector.driver.session() as session:
            session.run(query, edges=transformed_edges)
            logger.info("Edges inserted successfully.")
    except Exception as e:
        logger.error(f"Error inserting edges: {e}")
