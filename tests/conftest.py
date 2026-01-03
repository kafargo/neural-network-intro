"""
conftest.py
~~~~~~~~~~~

Pytest configuration and shared fixtures for all tests.
"""

import pytest
import numpy as np
import sys
import os
from io import BytesIO

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network import Network
from api_server import app, socketio, active_networks, training_jobs
import mnist_loader


@pytest.fixture
def simple_network():
    """Create a simple 3-layer network for testing."""
    return Network([3, 4, 2])


@pytest.fixture
def mnist_network():
    """Create a standard MNIST network."""
    return Network([784, 30, 10])


@pytest.fixture
def sample_training_data():
    """Create small sample training data for testing."""
    # Create 10 simple training examples
    training_data = []
    for i in range(10):
        x = np.random.randn(3, 1)
        y = np.zeros((2, 1))
        y[i % 2] = 1.0
        training_data.append((x, y))
    return training_data


@pytest.fixture
def sample_test_data():
    """Create small sample test data for testing."""
    # Create 5 simple test examples
    test_data = []
    for i in range(5):
        x = np.random.randn(3, 1)
        y = i % 2
        test_data.append((x, y))
    return test_data


@pytest.fixture
def flask_client():
    """Create a Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
    # Cleanup after test
    active_networks.clear()
    training_jobs.clear()


@pytest.fixture
def socketio_client():
    """Create a SocketIO test client."""
    client = socketio.test_client(app)
    yield client
    client.disconnect()
    # Cleanup after test
    active_networks.clear()
    training_jobs.clear()


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir)


@pytest.fixture(scope="session")
def mnist_data():
    """Load MNIST data once for all tests (slow operation)."""
    try:
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        return training_data, validation_data, test_data
    except Exception as e:
        pytest.skip(f"MNIST data not available: {e}")

