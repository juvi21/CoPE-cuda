import pytest
import torch
import cumsum

@pytest.fixture
def input_tensor():
    return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).cuda()

@pytest.fixture
def expected_output_cumsum():
    return torch.tensor([[1, 3, 6], [4, 9, 15]], dtype=torch.float32).cuda()

def test_cumsum(input_tensor, expected_output_cumsum):
    dim = 1
    output_tensor = cumsum.cumsum(input_tensor, dim)
    assert torch.allclose(output_tensor, expected_output_cumsum), f"Expected {expected_output_cumsum} but got {output_tensor}"
