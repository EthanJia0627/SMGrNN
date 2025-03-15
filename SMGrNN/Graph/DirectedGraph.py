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
    
    def draw(self):
        data = self.to_data()
        G = torch_geometric.utils.to_networkx(data, to_undirected=False)
        # Input nodes in Green, Output nodes in Red, Hidden nodes in Blue
        colors = ['g' if x == 0 else 'r' if x == 1 else 'b' for x in self.node_types]
        plt.figure()
        # Draw nodes
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors, node_size=500)
        # Draw edges
        nx.draw_networkx_edges(G, pos=pos)
        # Draw labels
        labels = {i: i for i in range(self.nodes.size(0))}
        nx.draw_networkx_labels(G, pos=pos, labels=labels)