import h5py

def read_nodes(file_path:str):
    with h5py.File(file_path, 'r') as f:
        nodes_group = f['/nodes/NodeA/0']
        node_data = {}

        # Extract each dataset
        for dataset_name in nodes_group:
            dataset = nodes_group[dataset_name]
            node_data[dataset_name] = dataset[()]

        return node_data

def read_edges(file_path:str):
    with h5py.File(file_path, 'r') as f:
        edges_group = f['/edges/NodeA__NodeB__chemical/0']
        edge_data = {}

        # Extract each dataset
        for dataset_name in edges_group:
            dataset = edges_group[dataset_name]
            edge_data[dataset_name] = dataset[()]

        return edge_data