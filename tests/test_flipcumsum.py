import pytest
import torch
import flip
import cumsum

@pytest.fixture
def input_tensor():
    return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).cuda()

@pytest.fixture
def expected_output_flipcumsum():
    return torch.tensor([[6, 5, 3], [15, 11, 6]], dtype=torch.float32).cuda()

def test_flipcumsum(input_tensor, expected_output_flipcumsum):
    flipped_tensor = flip.flip(input_tensor, 1)
    cumsum_tensor = cumsum.cumsum(flipped_tensor, 1)
    output_tensor = flip.flip(cumsum_tensor, 1)
    
    assert torch.allclose(output_tensor, expected_output_flipcumsum), f"Expected {expected_output_flipcumsum} but got {output_tensor}"
