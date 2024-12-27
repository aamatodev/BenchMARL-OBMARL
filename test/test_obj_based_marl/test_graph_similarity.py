import torch
import pytest


def graph_distance(objective_node_features, agent_node_features):
    graph_dist = []
    for i in range(objective_node_features.shape[0]):
        env_dist = []
        for j in range(objective_node_features.shape[1]):
            distance = torch.min((torch.linalg.norm(objective_node_features[i][j] - agent_node_features[i], dim=1))) / 8
            env_dist.append(distance)
        graph_dist.append(torch.sum(torch.stack(env_dist)) / 4)
    return 1 - torch.stack(graph_dist)


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


def test_graph_distance_cases():
    # Different tensors
    objective_node_features_diff = torch.rand((2, 4, 16))
    agent_node_features_diff = torch.rand((2, 4, 16))
    output_diff = graph_distance(objective_node_features_diff, agent_node_features_diff)
    assert output_diff.shape == (2,), f"Expected output shape (2,), but got {output_diff.shape}"

    # Equal tensors
    objective_node_features_equal = torch.tensor([[[3.0, 4.0], [1.0, 2.0]], [[7.0, 8.0], [5.0, 6.0]]])
    agent_node_features_equal = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    expected_output_equal = torch.tensor([1.0, 1.0])
    output_equal = graph_distance(objective_node_features_equal, agent_node_features_equal)
    assert torch.allclose(output_equal, expected_output_equal, atol=1e-4), f"Expected {expected_output_equal}, but got {output_equal}"

    # Partially equal tensors
    objective_node_features_partial = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    agent_node_features_partial = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[7.0, 8.0], [5.0, 6.0]]])
    expected_output_partial = torch.tensor([0.0, 0.0])
    output_partial = graph_distance(objective_node_features_partial, agent_node_features_partial)
    assert torch.allclose(output_partial, expected_output_partial, atol=1e-4), f"Expected {expected_output_partial}, but got {output_partial}"

# Run the test
pytest.main([__file__])

# Run the test

pytest.main([__file__])