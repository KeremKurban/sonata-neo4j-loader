import h5py

def read_nodes(file_path):
    with h5py.File(file_path, 'r') as f:
        nodes = f['/nodes']
        # Extract node properties
        return nodes

def read_edges(file_path):
    with h5py.File(file_path, 'r') as f:
        edges = f['/edges']
        # Extract edge properties
        return edges