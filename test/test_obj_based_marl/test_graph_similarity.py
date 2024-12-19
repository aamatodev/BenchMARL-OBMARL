import torch
import pytest


def graph_distance(objective_node_features, agent_node_features):
    graph_dist = []
    for i in range(objective_node_features.shape[0]):
        env_dist = []
        for j in range(objective_node_features.shape[1]):
            distance = torch.min(torch.linalg.norm(objective_node_features[i][j] - agent_node_features[i], dim=1))
            env_dist.append(distance)
        graph_dist.append(torch.sum(torch.stack(env_dist)))
    return torch.stack(graph_dist)


def test_graph_distance():
    # Sample input tensors
    objective_node_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    agent_node_features = torch.tensor([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])

    # Expected output tensor
    expected_output = torch.tensor([3.2361, 7.2361])

    # Call the function
    output = graph_distance(objective_node_features, agent_node_features)

    # Assert the output
    assert torch.allclose(output, expected_output, atol=1e-4), f"Expected {expected_output}, but got {output}"


def test_graph_distance_equal_graphs_inverted_order():
    # Sample input tensors where both graphs are equal but with inverted node order
    objective_node_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    agent_node_features = torch.tensor([[[3.0, 4.0], [1.0, 2.0]], [[7.0, 8.0], [5.0, 6.0]]])

    # Expected output tensor
    expected_output = torch.tensor([0.0, 0.0])

    # Call the function
    output = graph_distance(objective_node_features, agent_node_features)

    # Assert the output
    assert torch.allclose(output, expected_output, atol=1e-4), f"Expected {expected_output}, but got {output}"


# Run the test
pytest.main([__file__])