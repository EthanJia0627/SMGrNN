from matplotlib import pyplot as plt
import torch
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import networkx as nx
from torch_geometric.utils import degree,k_hop_subgraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SMGrNN(MessagePassing):
    def __init__(self, num_input_node, num_hidden_node, num_output_node,num_features,density=0.1,activation=torch.nn.functional.relu,edge_dict=None):
        super().__init__(aggr="add")
        self.num_input_node = num_input_node
        self.num_hidden_node = num_hidden_node
        self.num_output_node = num_output_node
        self.num_features = num_features
        self.density = density
        self.num_nodes = self.num_input_node + self.num_hidden_node + self.num_output_node
        self.activation = activation
        self.generate_initial_graph(edge_dict)

    def generate_initial_graph(self,edge_dict=None):
        # generate initial graph
        if edge_dict is None:
            edge_dict = self.generate_edge_dict()
        edge_weight = self.generate_edge_weight(edge_dict)
        self.g = DirectedGraph(
            nodes=torch.zeros([self.num_nodes, self.num_features],device=device),
            edge_dict=edge_dict,
            edge_weight=edge_weight,
            num_input_nodes=self.num_input_node,
            num_output_nodes=self.num_output_node,
        )
        

    def generate_edge_dict(self):
        edge_dict = {}
        # input nodes can not have incoming edges
        for i in range(self.num_nodes):
            edge_dict[i] = []
            for j in range(self.num_nodes):
                if np.random.rand() < self.density and j>=self.num_input_node:
                    edge_dict[i].append(j)
        return edge_dict
    
    def generate_edge_weight(self,edge_dict):
        edge_weight = {}
        for i in edge_dict:
            edge_weight[i] = torch.zeros(len(edge_dict[i]),dtype=torch.float32,device=device)
            for j in range(len(edge_dict[i])):
                edge_weight[i][j] = torch.rand(1)
        return edge_weight

    def forward(self, inputs=None):
        data = self.g.to_data()
        with torch.no_grad():
            if inputs is not None:
                data.x[: self.g.num_input_nodes] = (
                    data.x[: self.g.num_input_nodes] * 0.0 + inputs
                )
        nodes = self.compute_propagation(data.x, data.edge_index, data.edge_weight)
        return nodes[torch.where(self.g.node_types == 1)], nodes

    def compute_propagation(self, x, edge_index, edge_weight):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        for node in torch.where(self.g.node_types != 0):
            x[node] = out[node]
        return x

    def message(self, x_j,x_i,edge_index,edge_weight=None):
        """
        x_j is the input node features of the neighbors: E x F
        x_i is the input node features of the current node: E x F
        edge_index is the edge index: 2 x E
        edge_weight is the edge weight: E x 1
        """
        ## TODO Implement edge weights during message passing 
        return x_j * edge_weight.view(-1,1)

    def update(self, aggr_out):
        return self.activation(aggr_out)


        



class DirectedGraph:
    def __init__(self, nodes, edge_dict, edge_weight, num_input_nodes, num_output_nodes):
            self.nodes = nodes
            self.edge_dict = edge_dict
            self.edge_weight = edge_weight
            self.num_input_nodes = num_input_nodes
            self.num_hidden_nodes = nodes.size(0) - num_input_nodes - num_output_nodes
            self.num_output_nodes = num_output_nodes
            self.node_types = torch.zeros(self.nodes.size(0),device=self.nodes.device)
            # Node types: 0 - input, 1 - output, 2 - hidden
            self.node_types[:self.num_input_nodes] = 0                                              # input nodes
            self.node_types[self.num_input_nodes:self.num_input_nodes+self.num_output_nodes] = 1    # output nodes
            self.node_types[self.num_input_nodes+self.num_output_nodes:] = 2                        # hidden nodes

    def add_node(self, nodes, node_types):
        self.nodes = torch.cat([self.nodes, nodes], dim=0)
        self.node_types = torch.cat([self.node_types, node_types], dim=0)
        self.num_input_nodes += torch.sum(node_types == 0).item()
        self.num_output_nodes += torch.sum(node_types == 1).item()
        self.num_hidden_nodes += torch.sum(node_types == 2).item()
        


    def add_edge(self, i, j):
        if i not in self.edge_dict:
            self.edge_dict[i] = []
        if j not in self.edge_dict[i]:
            self.edge_dict[i].append(j)
            self.edge_weight[i].append(torch.zeros(1,dtype=torch.float32,device=self.nodes.device))
    def to_data(self):
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
        # return node indices of input nodes and node features of input nodes
        return torch.where(self.node_types == 0)[0], self.nodes[self.node_types == 0]
    
    def output_nodes(self):
        # return node indices of output nodes and node features of output nodes
        return torch.where(self.node_types == 1)[0], self.nodes[self.node_types == 1]
    
    def hidden_nodes(self):
        # return node indices of hidden nodes and node features of hidden nodes
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



class HebbianOptimizer:
    ...

if __name__ == "__main__":
    num_input_node = 2
    num_hidden_node = 3
    num_output_node = 2
    num_features = 2
    density = 0.5
    model = SMGrNN(num_input_node, num_hidden_node, num_output_node,num_features,density)
    model.generate_initial_graph()
    data = model.g.to_data()
    print("Nodes:",model.g.nodes)
    print("Edge dict:",model.g.edge_dict)
    print("Input nodes:\n",model.g.input_nodes()[1])
    print("Output nodes:\n",model.g.output_nodes()[1])
    print("Hidden nodes:\n",model.g.hidden_nodes()[1])
    model.g.draw()
    plt.show()
    # Test Propagation
    inputs = torch.randint(5,[num_input_node,num_features],device=device).float()
    outputs, nodes = model.forward(inputs)
    print("Input nodes:\n",inputs)
    print("Outputs:\n",outputs)
    print("Nodes:\n",nodes)
    model.g.draw()
    plt.show()