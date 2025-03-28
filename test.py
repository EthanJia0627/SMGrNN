import torch
import numpy as np
from matplotlib import pyplot as plt
from SMGrNN.network import SMGrNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def test_propagation():
    """
    Test the forward propagation of the SMGrNN model
    """
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
    inputs = torch.randint(5,[num_input_node,num_features],device=device).float()-2
    outputs, nodes = model.forward(inputs)
    print("Input nodes:\n",inputs)
    print("Outputs:\n",outputs)
    print("Nodes:\n",nodes)
    print("Weights:\n",model.g.edge_weight)
    model.g.draw(type="NodeActivity",edge_weight=True)
    plt.show()

def test_gradient_train():
    """
    Test the gradient training
    """
    num_input_node = 2
    num_hidden_node = 3
    num_output_node = 2
    num_features = 2
    density = 0.5
    model = SMGrNN(num_input_node, num_hidden_node, num_output_node,num_features,density,device=device)
    # Test Gradient
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    for i in range(1000):
        optimizer.zero_grad()
        inputs = torch.randint(2,[num_input_node,num_features],device=device).float()-1
        outputs, nodes = model.forward(inputs)
        target = -inputs.clone()
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            model.visualize(type="NodeActivity",edge_weight=True)
            print("Loss:",loss.item())
    print("Input nodes:\n",inputs)
    print("Outputs:\n",outputs)
    print("Nodes:\n",nodes)
    print("Initial Weights:\n",model.g.edge_weight)
    model.visualize(type="NodeActivity",edge_weight=True)
    print("Sync Weights:\n",model.g.edge_weight)
    plt.show()

def test_shared_graph():
    """
    Test if the graph is shared between the model and the optimizer
    """
    num_input_node = 2
    num_hidden_node = 1
    num_output_node = 2
    num_features = 1
    density = 0.5
    model = SMGrNN(num_input_node, num_hidden_node, num_output_node,num_features,density,device=device)
    print("Nodes:",model.g.nodes)
    model.g.add_node(torch.tensor([[0],[1],[2]],device=device),torch.tensor([1,1,1],device=device))
    print("Nodes from Optimizer:",model.hebbian.g.nodes)

def test_growth():
    """
    Fit a XOR Gate with the network
    """
    num_input_node = 2
    num_hidden_node = 3
    num_output_node = 1
    num_features = 2
    density = 0.5
    model = SMGrNN(num_input_node, num_hidden_node, num_output_node,num_features,density,device=device)
    # Train the network
    for epoch in range(1000):
        inputs = torch.randint(0,2,[num_input_node,num_features],device=device)
        # Calculate the XOR gate
        target = inputs[0]^inputs[1]
        target = target.unsqueeze(0)
        loss = model.train_step(inputs.float(),target.float())
        if epoch%100 == 0:
            model.grow()
            model.visualize("NodeActivity",edge_weight=True)
            print("Epoch:",epoch)
            print("Loss:",loss)
    


    
if __name__ == "__main__":
    # test_propagation()
    # test_gradient_train()
    # test_shared_graph()
    test_growth()