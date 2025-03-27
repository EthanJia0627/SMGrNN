import torch
import numpy as np
import networkx as nx
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.data import Data



class DirectedGraph:
    def __init__(self, nodes, edge_dict, edge_weight, num_input_nodes, num_output_nodes):
        """
        Initialize the DirectedGraph
        """
        # ========================= Node =========================
        self.nodes = nodes
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = nodes.size(0) - num_input_nodes - num_output_nodes
        self.num_output_nodes = num_output_nodes
        self.node_types = torch.zeros(self.nodes.size(0),device=self.nodes.device)              # Node types: 0 - input, 1 - output, 2 - hidden
        self.node_types[:self.num_input_nodes] = 0                                              # input nodes
        self.node_types[self.num_input_nodes:self.num_input_nodes+self.num_output_nodes] = 1    # output nodes
        self.node_types[self.num_input_nodes+self.num_output_nodes:] = 2                        # hidden nodes
        # ========================= Edge =========================
        self.edge_dict = edge_dict
        self.edge_weight = edge_weight

    def add_node(self, nodes, node_types):
        """
        Add a node to the graph
        nodes: Node features (Tensor)
        node_types: Node types (Tensor)
        """
        # ========================= Node =========================
        self.nodes = torch.cat([self.nodes, nodes], dim=0) # Vertically concatenate the nodes
        self.num_input_nodes += torch.sum(node_types == 0).item()
        self.num_output_nodes += torch.sum(node_types == 1).item()
        self.num_hidden_nodes += torch.sum(node_types == 2).item()

        # ========================= Type =========================
        self.node_types = torch.cat([self.node_types, node_types], dim=0)

        
    def add_edge(self, i, j):
        """
        Add an edge to the graph
        i: Source node index (Integer)
        j: Destination node index (Integer)
        """
        if ([i,j]>self.nodes.size(0)).any():
            raise ValueError("Node index out of range")
        if i not in self.edge_dict:
            self.edge_dict[i] = []
        if j not in self.edge_dict[i]:
            self.edge_dict[i].append(j)
            self.edge_weight[i].append(torch.zeros(1,dtype=torch.float32,device=self.nodes.device))

    def to_data(self):
        """
        Convert the graph to PyG Data 
        DataStructure: 
            x: Node features (N x F)
            edge_index: Edge index (2 x E)
            edge_weight: Edge weight (E)
        """
        edges = []
        weights = []
        for node in self.edge_dict:
            destinations = self.edge_dict[node]
            for d in destinations:
                edges.append([node, d])
                weights.append(self.edge_weight[node][destinations.index(d)])

        edges = torch.tensor(edges).long().t().contiguous().to(self.nodes.device)
        weights = torch.tensor(weights).float().to(self.nodes.device)

        return Data(
            x=self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device),
            edge_index=edges,
            edge_weight=weights,
        )
    
    def sync_edge_weight(self, edge_weight):
        """
        Synchronize the edge weight with the provided edge weight
        edge_weight: Edge weight (Tensor) torch.Size([E])
        """
        edge_weight_flat = edge_weight.detach().view(-1)
        # Count total number of edges
        total_edges = 0
        for node in self.edge_dict:
            total_edges += len(self.edge_dict[node])

        if total_edges != len(edge_weight_flat):
            raise ValueError(f"Edge weight length mismatch: {total_edges} edges, but got {len(edge_weight_flat)} weights")

        idx = 0
        for node in self.edge_dict:
            destinations = self.edge_dict[node]
            for d in destinations:
                # Update the weight for this edge
                self.edge_weight[node][destinations.index(d)] = edge_weight_flat[idx].view(1).to(self.nodes.device)
                idx += 1


    def input_nodes(self):
        """
        return node indices of input nodes and node features of input nodes
        Returns:
            input_nodes: Node indices of input nodes (Ni)
            input_nodes_features: Node features of input nodes (Ni x F)
        """
        return torch.where(self.node_types == 0)[0], self.nodes[self.node_types == 0]
    
    def output_nodes(self):
        """
        return node indices of output nodes and node features of output nodes
        Returns:
            output_nodes: Node indices of output nodes (No)
            output_nodes_features: Node features of output nodes (No x F)
        """
        return torch.where(self.node_types == 1)[0], self.nodes[self.node_types == 1]
    
    def hidden_nodes(self):
        """
        return node indices of hidden nodes and node features of hidden nodes
        Returns:
            hidden_nodes: Node indices of hidden nodes (Nh)
            hidden_nodes_features: Node features of hidden nodes (Nh x F)
        """
        return torch.where(self.node_types == 2)[0], self.nodes[self.node_types == 2]
    
    def draw(self,type = "NodeType",edge_weight = False):
        """
        Draw the graph
        type: Type of the graph (String) - "NodeType"  or "NodeActivity"
        edge_weight: Whether to draw the edge weight (Boolean)
        """
        data = self.to_data()
        G = torch_geometric.utils.to_networkx(data, to_undirected=False)

        # =========================================== Fix Layout ===========================================
        if not hasattr(self, '_pos_cache'):
            self._pos_cache = {}
        # Create a hashable key based on current edge structure
        edge_key = tuple(sorted((int(i), int(j)) for i, j in zip(*data.edge_index.cpu().numpy())))
        if edge_key in self._pos_cache:
            pos = self._pos_cache[edge_key]
        else:
            pos = nx.kamada_kawai_layout(G)  # or nx.spectral_layout(G)
            self._pos_cache[edge_key] = pos

        # ======================================================================================================================
        # ===================================================== Draw Graph =====================================================
        # ======================================================================================================================
        if edge_weight:
            # Edge weight in cyan if positive and magenta if negative, width is proportional to the absolute value of the weight
            edge_colors = [(1,0,0,1) if x > 0 else (0.1,0.25,0.8,1) for x in data.edge_weight]
            edge_width = [abs(x) for x in data.edge_weight]

        # ======================================== Node Type Graph ========================================
        if type == "NodeType":
            # Input nodes in Green, Output nodes in Red, Hidden nodes in Blue
            colors = ['g' if x == 0 else 'r' if x == 1 else 'b' for x in self.node_types]
            plt.figure()
            if edge_weight:
                # Draw edges
                nx.draw(G, pos=pos, with_labels=True, node_color=colors, edge_color=edge_colors, width=edge_width)
            else:
                # Draw nodes
                nx.draw(G, pos=pos, with_labels=True, node_color=colors)

        # ====================================== Node Activity Graph ======================================
        elif type == "NodeActivity":
            # Node activity in magenta(RGBA) if positive and cyan(RGBA) if negative, density is proportional to the absolute value of the activity
            # Sum features for each node to get activity values
            node_activities = torch.sum(self.nodes, dim=1) if len(self.nodes.shape) > 1 else self.nodes
            
            # Get maximum absolute activity for normalization
            max_abs_activity = torch.abs(node_activities).max().item()
            if max_abs_activity > 0:  # Avoid division by zero
                normalized_abs_activity = torch.abs(node_activities) / max_abs_activity
            else:
                normalized_abs_activity = torch.zeros_like(node_activities)

            # Create colors list with appropriate RGBA values
            colors = []
            for i in range(len(node_activities)):
                activity = node_activities[i].item()
                intensity = normalized_abs_activity[i].item()
                
                if activity > 0:  # Positive activity - magenta
                    # Interpolate between white (1,1,1) and magenta (1,0,1)
                    colors.append((np.array([1, 1, 1, 0]) - np.pow(intensity,1/4) * np.array([0, 1, 1, -1])).tolist())
                elif activity < 0:  # Negative activity - cyan
                    # Interpolate between white (1,1,1) and cyan (0,1,1)
                    colors.append((np.array([1, 1, 1, 1]) - np.pow(intensity,1/4) * np.array([1, 0.75, 0.2, 0])).tolist())
                else:  # Zero activity - white
                    colors.append((1, 1, 1, 1))
            plt.figure()
            if edge_weight:
                # Draw edges
                nx.draw(G, pos=pos, with_labels=True, node_color=colors, edge_color=edge_colors, 
                       width=edge_width, edgecolors='black', linewidths=0.2)
            else:
                # Draw nodes
                nx.draw(G, pos=pos, with_labels=True, node_color=colors, 
                       edgecolors='black', linewidths=0.2)
                
