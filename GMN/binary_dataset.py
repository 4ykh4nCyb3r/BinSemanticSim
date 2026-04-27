import json
import numpy as np
import collections
import random
import torch

debug = True

# The exact object format the Graph Matching Network expects
GraphData = collections.namedtuple('GraphData', [
    'from_idx', 'to_idx', 'node_features', 'edge_features', 'graph_idx', 'n_graphs'])

class BinaryDataset:
    def __init__(self, json_file_1, json_file_2):
        print(f"[*] Loading AI Dataset from {json_file_1} and {json_file_2}...")
        with open(json_file_1, 'r') as f:
            funcs_v1 = json.load(f)
        with open(json_file_2, 'r') as f:
            funcs_v2 = json.load(f)
        
        # Parse the JSON into math-ready graphs
        self.graphs_v1 = self._parse_functions(funcs_v1)
        exit(0) # Debugging: Stop after parsing the first file to check the output
        self.graphs_v2 = self._parse_functions(funcs_v2)
        
        # Find functions that exist in both versions (so we can compare them)
        self.common_funcs = list(set(self.graphs_v1.keys()).intersection(set(self.graphs_v2.keys())))
        print(f"[*] Found {len(self.common_funcs)} matching functions to train on.")

    def _parse_functions(self, functions):
        parsed_graphs = {}
        for func in functions:
            name = func['function_name']
            blocks = func.get('blocks', [])
            edges = func.get('edges', [])
            if debug:
                print(f"Parsing function: {name} with {len(blocks)} blocks and {len(edges)} edges.")
                print("Blocks:")
                print(blocks)
                print()
                print("Edges:")
                print(edges)
                break # Only parse the first function for debugging purposes
               

            if len(blocks) == 0:
                continue
            
            # Map hex addresses to 0, 1, 2, ...
            addr_to_idx = {block['address']: i for i, block in enumerate(blocks)}
            
            # 1. Node Features (The Bag-of-Words math we just generated)
            node_features = np.array([block['features'] for block in blocks], dtype=np.float32)
            
            # 2. Edges
            from_idx, to_idx = [], []
            for src, dst in edges:
                if src in addr_to_idx and dst in addr_to_idx:
                    from_idx.append(addr_to_idx[src])
                    to_idx.append(addr_to_idx[dst])
            
            # 3. Dummy Edge Features (Since angr doesn't give edge types, we use 1s)
            edge_features = np.ones((len(from_idx), 1), dtype=np.float32)
            
            parsed_graphs[name] = {
                'node_features': node_features,
                'edge_features': edge_features,
                'from_idx': np.array(from_idx, dtype=np.int32),
                'to_idx': np.array(to_idx, dtype=np.int32)
            }
        return parsed_graphs

    def _pack_batch(self, graphs):
        """Batches multiple graphs together into one giant graph for the GPU to process fast."""
        from_idx, to_idx, node_features, edge_features, graph_idx = [], [], [], [], []
        node_offset = 0

        for i, g in enumerate(graphs):
            n_nodes = g['node_features'].shape[0]
            node_features.append(g['node_features'])
            edge_features.append(g['edge_features'])
            from_idx.append(g['from_idx'] + node_offset)
            to_idx.append(g['to_idx'] + node_offset)
            graph_idx.append(np.full(n_nodes, i, dtype=np.int32))
            node_offset += n_nodes

        return GraphData(
            from_idx=np.concatenate(from_idx) if from_idx else np.array([], dtype=np.int32),
            to_idx=np.concatenate(to_idx) if to_idx else np.array([], dtype=np.int32),
            node_features=np.concatenate(node_features),
            edge_features=np.concatenate(edge_features) if edge_features else np.zeros((0, 1), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx),
            n_graphs=len(graphs)
        )

    def pairs(self, batch_size):
        """The core training engine: Generates flashcards of Pairs (Function V1 vs Function V2)"""
        while True:
            batch_graphs = []
            labels = []
            for _ in range(batch_size):
                # 50% chance to generate a Clone (Positive), 50% chance for Non-Clone (Negative)
                is_positive = random.random() > 0.5
                
                if is_positive:
                    func_name = random.choice(self.common_funcs)
                    # Compare 'main' from v1 to 'main' from v2
                    batch_graphs.extend([self.graphs_v1[func_name], self.graphs_v2[func_name]])
                    labels.append(1)  # 1 = Clone
                else:
                    func_1 = random.choice(self.common_funcs)
                    func_2 = random.choice(self.common_funcs)
                    while func_1 == func_2: 
                        func_2 = random.choice(self.common_funcs)
                    # Compare 'main' from v1 to 'process_data' from v2
                    batch_graphs.extend([self.graphs_v1[func_1], self.graphs_v2[func_2]])
                    labels.append(-1) # -1 = Not a Clone

            #yield self._pack_batch(batch_graphs), torch.tensor(labels, dtype=np.float32)
            #yield self._pack_batch(batch_graphs), torch.tensor(labels, dtype=torch.float32)
            yield self._pack_batch(batch_graphs), np.array(labels, dtype=np.int32)