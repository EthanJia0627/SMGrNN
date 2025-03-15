import torch
import numpy as np
from matplotlib import pyplot as plt
from SMGrNN.network import SMGrNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_input_node = 2
num_hidden_node = 3
num_output_node = 2
num_features = 2
density = 0.5
model = SMGrNN(num_input_node, num_hidden_node, num_output_node,num_features,density,device=device)
# Test Graph Generation
model.generate_initial_graph()
data = model.g.to_data()
print("Nodes:",model.g.nodes)
print("Edge dict:",model.g.edge_dict)
print("Input nodes:\n",model.g.input_nodes()[1])
print("Output nodes:\n",model.g.output_nodes()[1])
print("Hidden nodes:\n",model.g.hidden_nodes()[1])
model.g.draw(type="NodeActivity",edge_weight=True)
plt.show()
# Test Propagation
inputs = torch.randint(5,[num_input_node,num_features],device=device).float()
outputs, nodes = model.forward(inputs)
print("Input nodes:\n",inputs)
print("Outputs:\n",outputs)
print("Nodes:\n",nodes)
print("Weights:\n",model.g.edge_weight)
model.g.draw(type="NodeActivity",edge_weight=True)
plt.show()