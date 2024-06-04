import pytest
import torch
import flip

@pytest.fixture
def input_tensor():
    return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).cuda()

@pytest.fixture
def expected_output_flip():
    return torch.tensor([[3, 2, 1], [6, 5, 4]], dtype=torch.float32).cuda()

def test_flip(input_tensor, expected_output_flip):
    dim = 1
    output_tensor = flip.flip(input_tensor, dim)
    assert torch.allclose(output_tensor, expected_output_flip), f"Expected {expected_output_flip} but got {output_tensor}"
