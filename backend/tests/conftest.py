import pytest
import numpy as np

@pytest.fixture
def small_grid_size():
    """Small grid for fast tests."""
    return 10

@pytest.fixture
def default_grid_size():
    """Standard grid size."""
    return 50
