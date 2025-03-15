import torch
import numpy as np
from matplotlib import pyplot as plt
from .Graph.DirectedGraph import DirectedGraph
from torch_geometric.nn import MessagePassing

class SMGrNN(MessagePassing):
    def __init__(self, num_input_node, num_hidden_node, num_output_node,num_features,density=0.1,activation=torch.nn.functional.tanh,edge_dict=None,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__(aggr="add")
        """
        Initialize the SMGrNN model with graph generation
        num_input_node: Number of input nodes (Integer)
        num_hidden_node: Number of hidden nodes (Integer)
        num_output_node: Number of output nodes (Integer)
        num_features: Number of features (Integer)
        density: Density of the graph (Float)
        activation: Activation function (torch.nn.functional)
        edge_dict: Dictionary of edges (Dict)
        """
        # ========================= Device  =========================
        self.device = device
        self.to(device)
        # ========================= Node  =========================
        self.num_input_node = num_input_node
        self.num_hidden_node = num_hidden_node
        self.num_output_node = num_output_node
        self.num_nodes = self.num_input_node + self.num_hidden_node + self.num_output_node
        self.num_features = num_features
        # ========================= Graph  =========================
        self.g = None
        self.density = density
        self.activation = activation
        self.generate_initial_graph(edge_dict)


    def generate_initial_graph(self,edge_dict=None):
        """
        Generate the initial graph with random edges or given edge_dict
        edge_dict: Dictionary of edges (Dict)
        """
        # generate initial graph
        if edge_dict is None:
            edge_dict = self.generate_edge_dict()
        edge_weight = self.generate_edge_weight(edge_dict)
        self.g = DirectedGraph(
            nodes=torch.zeros([self.num_nodes, self.num_features],device=self.device),
            edge_dict=edge_dict,
            edge_weight=edge_weight,
            num_input_nodes=self.num_input_node,
            num_output_nodes=self.num_output_node,
        )
        
    def generate_edge_dict(self):
        """
        Generate the edge dictionary for the graph
        """
        edge_dict = {}
        # input nodes can not have incoming edges
        for i in range(self.num_nodes):
            edge_dict[i] = []
            for j in range(self.num_nodes):
                if np.random.rand() < self.density and j>=self.num_input_node:
                    edge_dict[i].append(j)
        return edge_dict
    
    def generate_edge_weight(self,edge_dict):
        """
        Generate the edge weight for the graph
        edge_dict: Dictionary of edges (Dict)
        """
        edge_weight = {}
        for i in edge_dict:
            edge_weight[i] = torch.zeros(len(edge_dict[i]),dtype=torch.float32,device=self.device)
            for j in range(len(edge_dict[i])):
                edge_weight[i][j] = 2*torch.rand(1)-1
        return edge_weight

    def forward(self, inputs=None):
        """"
        Forward pass of the model
        inputs: Input features (Tensor)
        """
        data = self.g.to_data()
        # Set input nodes to the input features
        with torch.no_grad():
            if inputs is not None:
                data.x[: self.g.num_input_nodes] = (
                    data.x[: self.g.num_input_nodes] * 0.0 + inputs
                )
        nodes = self.compute_propagation(data.x, data.edge_index, data.edge_weight)
        self.g.nodes = nodes
        return nodes[torch.where(self.g.node_types == 1)], nodes

    def compute_propagation(self, x, edge_index, edge_weight):
        """
        Compute the propagation of the graph
        x: Node features (Tensor)
        edge_index: Edge index (Tensor)
        edge_weight: Edge weight (Tensor)
        """
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
        # TODO Compute new weights according to in and out degrees
        # 
        return x_j * edge_weight.view(-1,1)

    def update(self, aggr_out):
        """
        aggr_out is the aggregated message: N x F
        """
        return self.activation(aggr_out)


        


"""
Example Usage:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
"""
    