import logging
from typing import Any, Dict, List, Tuple
import pandas as pd
from bluepysnap import Circuit
import libsonata
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_nodes(
    circuit: Circuit, node_set: str, proportion: float
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract and subsample nodes from node populations.

    Parameters
    ----------
    circuit : Circuit
        An instance of the BluePySnap Circuit class.
    node_set: str
        Node set to be extracted from entire circuit.
    proportion : float
        The proportion of nodes to sample (between 0 and 1). If node set and proportion together is selected,
        joint probability will be retreived.

    Returns
    -------
    sampled_nodes_df : pd.DataFrame
        DataFrame containing sampled nodes with 'id' and 'population_name' columns.
    belongs_to_relations : list of dict
        A list of dictionaries for BELONGS_TO relationships.
    populations : list of dict
        A list of dictionaries containing population properties.
    """
    nodes_df_list = []
    belongs_to_relations = []
    populations = []
    for pop_name in circuit.nodes.population_names:
        logger.info(f"Extracting nodes from population: {pop_name}")
        node_population = circuit.nodes[pop_name]
        node_set_ids = node_population.ids(
            node_set
        ).tolist()  # can be non-contiguous ids
        node_storage = libsonata.NodeStorage(node_population.h5_filepath)
        population = node_storage.open_population(pop_name)
        # node_ids = list(range(population.size))
        selection = libsonata.Selection(node_set_ids)
        attr_types = list(population.attribute_names)
        attributes = {
            attr: population.get_attribute(attr, selection) for attr in attr_types
        }
        df = pd.DataFrame(attributes)
        df["id"] = node_set_ids
        df["population_name"] = pop_name
        nodes_df_list.append(df)
        belongs_to_relations.extend(
            [
                {"node_id": node_id, "population_name": pop_name}
                for node_id in node_set_ids
            ]
        )
        populations.append(
            {
                "name": pop_name,
                "size": population.size,
                "attr_types": attr_types,
            }
        )
    all_nodes_df = pd.concat(nodes_df_list, ignore_index=True)
    sampled_nodes_df = all_nodes_df.sample(frac=proportion, random_state=42)
    # Update belongs_to_relations to include only sampled nodes
    sampled_node_ids = set(
        zip(sampled_nodes_df["id"], sampled_nodes_df["population_name"])
    )
    belongs_to_relations = [
        rel
        for rel in belongs_to_relations
        if (rel["node_id"], rel["population_name"]) in sampled_node_ids
    ]
    return sampled_nodes_df, belongs_to_relations, populations


def extract_edges(
    circuit: Circuit, proportion: float, sampled_nodes_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Extract and subsample edges from edge populations.

    Parameters
    ----------
    circuit : Circuit
        An instance of the BluePySnap Circuit class.
    proportion : float
        The proportion of edges to sample (between 0 and 1).
    sampled_nodes_df : pd.DataFrame
        DataFrame containing sampled nodes with 'id' and 'population_name' columns.

    Returns
    -------
    edges : list of dict
        A list of dictionaries containing edge properties.
    """
    edges_df_list = []
    sampled_nodes_set = set(
        zip(sampled_nodes_df["id"], sampled_nodes_df["population_name"])
    )

    for pop_name in circuit.edges.population_names:
        logger.info(f"Extracting edges from population: {pop_name}")
        edge_population = circuit.edges[pop_name]
        edge_storage = libsonata.EdgeStorage(edge_population.h5_filepath)
        population = edge_storage.open_population(pop_name)
        logger.info(f"Total edges found in population {pop_name}: {population.size}")
        # Select a subset of edge IDs based on the given proportion
        total_edges = population.size
        selected_edge_count = int(total_edges * proportion)
        logger.info(
            f"Selected {selected_edge_count} edges for import from population {pop_name}."
        )

        user_input = (
            input(
                f"Do you want to proceed with importing {selected_edge_count} edges? (yes/no): "
            )
            .strip()
            .lower()
        )
        if user_input != "yes":
            logger.info("Edge import process terminated by user.")
            return []
        edge_ids = list(range(total_edges))
        selected_edge_ids = random.sample(edge_ids, selected_edge_count)

        selection = libsonata.Selection(selected_edge_ids)
        attr_types = list(population.attribute_names)
        attributes = {
            attr: population.get_attribute(attr, selection) for attr in attr_types
        }
        source_node_ids = [population.source_node(eid) for eid in selected_edge_ids]
        target_node_ids = [population.target_node(eid) for eid in selected_edge_ids]

        # Parse pop_name to get source and target population names
        pop_name_parts = pop_name.split("__")
        if len(pop_name_parts) >= 3:
            source_population_name = pop_name_parts[0]
            target_population_name = pop_name_parts[1]
        elif pop_name == "default":
            logging.warning(
                "WARNING: Edge pop name doesnt comply with SONATA standard."
            )
            source_population_name = "hippocampus_neurons"
            target_population_name = "hippocampus_neurons"
        else:
            logger.error(f"Unable to parse population names from pop_name: {pop_name}")
            continue

        df = pd.DataFrame(attributes)
        df["source_node_id"] = source_node_ids
        df["source_population_name"] = source_population_name
        df["target_node_id"] = target_node_ids
        df["target_population_name"] = target_population_name

        # Vectorized filtering of edges where both nodes are sampled
        source_nodes = list(zip(df["source_node_id"], df["source_population_name"]))
        target_nodes = list(zip(df["target_node_id"], df["target_population_name"]))
        mask = pd.Series(source_nodes).isin(sampled_nodes_set) & pd.Series(
            target_nodes
        ).isin(sampled_nodes_set)
        df = df[mask]
        edges_df_list.append(df)

    if edges_df_list:
        all_edges_df = pd.concat(edges_df_list, ignore_index=True)
        edges = all_edges_df.to_dict("records")
    else:
        edges = []
    return edges
