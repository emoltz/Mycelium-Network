import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx


# Define a simple neural network for each node.
class MyceliumNode(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyceliumNode, self).__init__()
        # A single linear layer (you might add more complexity as needed)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Use ReLU activation to simulate non-linear processing
        return torch.relu(self.fc(x))


def create_mycelium_network(num_nodes, connection_prob=0.1, vec_size=10):
    """
    Creates a graph where each node has its own neural network.
    - num_nodes: number of nodes in the network.
    - connection_prob: probability that any two nodes are connected (Erdős–Rényi model).
    - vec_size: size of the input/output vector each node processes.
    """
    # Create a random graph (nodes will mimic the hyphae connections in mycelium)
    G = nx.erdos_renyi_graph(num_nodes, connection_prob)

    # For each node, assign a neural model and an optimizer.
    for node in G.nodes:
        # Each node processes a vector of length vec_size.
        model = MyceliumNode(input_size=vec_size, output_size=vec_size)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        G.nodes[node]['model'] = model
        G.nodes[node]['optimizer'] = optimizer

    return G


def simulate_mycelium_network(G, num_iterations=100, vec_size=10):
    """
    Simulate the mycelium network for a number of iterations.
    Each node:
      1. Collects inputs from its neighbors.
      2. Processes the aggregated signal via its neural network.
      3. Updates its weights based on a self-supervised loss.
    """
    for iteration in range(num_iterations):
        # Dictionary to store each node's output during this iteration.
        node_outputs = {}

        # Forward pass: each node gathers inputs and computes its output.
        for node in G.nodes:
            # Gather outputs from neighboring nodes.
            neighbors = list(G.neighbors(node))
            if neighbors:
                # Aggregate the outputs from neighbors.
                # (For nodes that haven't produced an output yet, use a random vector.)
                aggregated_input = torch.zeros(vec_size)
                for neighbor in neighbors:
                    if neighbor in node_outputs:
                        aggregated_input += node_outputs[neighbor]
                    else:
                        aggregated_input += torch.randn(vec_size)
                aggregated_input /= max(len(neighbors), 1)
            else:
                # If the node has no neighbors, supply a random input.
                aggregated_input = torch.randn(vec_size)

            # Process the aggregated input through the node's neural model.
            model = G.nodes[node]['model']
            output = model(aggregated_input)
            node_outputs[node] = output

        # Backward pass: accumulate the loss from all nodes first.
        total_loss = 0.0
        for node in G.nodes:
            target = torch.zeros(vec_size)  # This is a placeholder target.
            model = G.nodes[node]['model']
            output = node_outputs[node]
            loss = torch.mean((output - target) ** 2)
            total_loss += loss

        # Zero gradients for all optimizers.
        for node in G.nodes:
            optimizer = G.nodes[node]['optimizer']
            optimizer.zero_grad()

        # Backward pass on the total loss.
        total_loss.backward()

        # Update each node's model.
        for node in G.nodes:
            optimizer = G.nodes[node]['optimizer']
            optimizer.step()

        avg_loss = total_loss.item() / G.number_of_nodes()
        print(f"Iteration {iteration}: Average Loss = {avg_loss:.4f}")


# --- Main execution ---
if __name__ == "__main__":
    # Create a network of 20 nodes with a 20% connection probability.
    mycelium_net = create_mycelium_network(num_nodes=20, connection_prob=0.2, vec_size=10)

    # Simulate the network for 50 iterations.
    simulate_mycelium_network(mycelium_net, num_iterations=50, vec_size=10)
