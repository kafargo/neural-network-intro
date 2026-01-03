# Testing Quick Reference

## Quick Commands

```bash
# Run all tests
pytest
# or
python run_tests.py

# Run specific categories
python run_tests.py unit        # Unit tests only (fastest)
python run_tests.py api         # API tests only
python run_tests.py integration # Integration tests
python run_tests.py fast        # All except slow tests

# Coverage report
python run_tests.py coverage    # Generates htmlcov/index.html

# Help
python run_tests.py help
```

## Test Files

- `tests/test_network.py` - 26 neural network unit tests
- `tests/test_model_persistence.py` - 13 model save/load tests  
- `tests/test_api.py` - 27 API endpoint tests
- `tests/test_socketio.py` - 5 WebSocket event tests

**Total: 71 tests**

## Run Specific Tests

```bash
# Single test file
pytest tests/test_network.py

# Single test class
pytest tests/test_network.py::TestNetwork

# Single test function
pytest tests/test_network.py::TestNetwork::test_network_initialization

# By marker
pytest -m unit
pytest -m api
pytest -m "not slow"
```

## Available Fixtures

From `tests/conftest.py`:

- `simple_network` - [3, 4, 2] network for fast tests
- `mnist_network` - [784, 30, 10] MNIST network
- `sample_training_data` - 10 training samples
- `sample_test_data` - 5 test samples
- `flask_client` - Flask test client
- `socketio_client` - SocketIO test client
- `temp_model_dir` - Temporary directory
- `mnist_data` - Full MNIST dataset

## Writing a Test

```python
# tests/test_network.py

@pytest.mark.unit
def test_my_feature(simple_network):
    """Test description."""
    # Arrange
    net = simple_network
    
    # Act
    result = net.some_method()
    
    # Assert
    assert result is not None
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.slow` - Slower tests

## Coverage

```bash
# Generate coverage report
python run_tests.py coverage

# View in browser
open htmlcov/index.html
```

## Current Status

✅ 71 tests implemented
✅ All tests passing
✅ Unit tests run in < 1 second
✅ Full suite runs in < 5 seconds

## Documentation

- `README.md` - Main testing section
- `TESTING_SETUP.md` - Detailed guide
- `pytest.ini` - Configuration
- `tests/conftest.py` - Fixture reference

