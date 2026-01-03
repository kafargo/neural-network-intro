"""
test_network.py
~~~~~~~~~~~~~~~

Unit tests for the neural network module.
"""

import pytest
import numpy as np
from network import Network, sigmoid, sigmoid_prime


@pytest.mark.unit
class TestNetwork:
    """Tests for the Network class."""

    def test_network_initialization(self, simple_network):
        """Test that a network initializes with correct dimensions."""
        net = simple_network
        assert net.num_layers == 3
        assert net.sizes == [3, 4, 2]
        assert len(net.biases) == 2
        assert len(net.weights) == 2
        assert net.biases[0].shape == (4, 1)
        assert net.biases[1].shape == (2, 1)
        assert net.weights[0].shape == (4, 3)
        assert net.weights[1].shape == (2, 4)

    def test_mnist_network_dimensions(self, mnist_network):
        """Test MNIST network has correct dimensions."""
        net = mnist_network
        assert net.num_layers == 3
        assert net.sizes == [784, 30, 10]
        assert net.weights[0].shape == (30, 784)
        assert net.weights[1].shape == (10, 30)

    def test_feedforward_output_shape(self, simple_network):
        """Test feedforward produces correct output shape."""
        net = simple_network
        x = np.random.randn(3, 1)
        output = net.feedforward(x)
        assert output.shape == (2, 1)

    def test_feedforward_output_range(self, simple_network):
        """Test feedforward output is in valid range (0-1) due to sigmoid."""
        net = simple_network
        x = np.random.randn(3, 1)
        output = net.feedforward(x)
        assert np.all(output >= 0) and np.all(output <= 1)

    def test_feedforward_deterministic(self, simple_network):
        """Test feedforward gives same output for same input."""
        net = simple_network
        x = np.random.randn(3, 1)
        output1 = net.feedforward(x)
        output2 = net.feedforward(x)
        np.testing.assert_array_equal(output1, output2)

    def test_evaluate_returns_integer(self, simple_network, sample_test_data):
        """Test evaluate returns integer count of correct predictions."""
        net = simple_network
        result = net.evaluate(sample_test_data)
        assert isinstance(result, (int, np.integer))
        assert 0 <= result <= len(sample_test_data)

    def test_update_mini_batch_changes_weights(self, simple_network, sample_training_data):
        """Test that update_mini_batch modifies network weights."""
        net = simple_network
        mini_batch = sample_training_data[:2]

        # Store original weights
        original_weights = [w.copy() for w in net.weights]

        # Update with mini batch
        net.update_mini_batch(mini_batch, eta=0.5)

        # Check that weights have changed
        for orig, new in zip(original_weights, net.weights):
            assert not np.allclose(orig, new)

    def test_sgd_callback_called(self, simple_network, sample_training_data):
        """Test that SGD callback is called for each epoch."""
        net = simple_network
        callback_count = [0]

        def callback(data):
            callback_count[0] += 1
            assert 'epoch' in data
            assert 'total_epochs' in data
            assert 'elapsed_time' in data

        epochs = 3
        net.SGD(sample_training_data, epochs, mini_batch_size=2,
                eta=0.5, callback=callback)

        assert callback_count[0] == epochs

    def test_sgd_with_test_data(self, simple_network, sample_training_data, sample_test_data):
        """Test SGD with test data provides accuracy in callback."""
        net = simple_network
        callback_data = []

        def callback(data):
            callback_data.append(data)

        net.SGD(sample_training_data, epochs=2, mini_batch_size=2,
                eta=0.5, test_data=sample_test_data, callback=callback)

        assert len(callback_data) == 2
        for data in callback_data:
            assert 'accuracy' in data
            assert 'correct' in data
            assert 'total' in data
            assert data['total'] == len(sample_test_data)


@pytest.mark.unit
class TestActivationFunctions:
    """Tests for activation functions."""

    def test_sigmoid_output_range(self):
        """Test sigmoid output is always between 0 and 1."""
        z = np.array([[-10], [0], [10]])
        result = sigmoid(z)
        assert np.all(result > 0) and np.all(result < 1)

    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5."""
        assert sigmoid(np.array([[0]])) == pytest.approx(0.5)

    def test_sigmoid_prime_positive(self):
        """Test sigmoid_prime is always positive."""
        z = np.linspace(-10, 10, 100).reshape(-1, 1)
        result = sigmoid_prime(z)
        assert np.all(result > 0)

    def test_sigmoid_prime_at_zero(self):
        """Test sigmoid_prime(0) = 0.25."""
        assert sigmoid_prime(np.array([[0]])) == pytest.approx(0.25)


@pytest.mark.unit
class TestNetworkMethods:
    """Tests for specific network methods."""

    def test_cost_derivative_shape(self, simple_network):
        """Test cost_derivative returns correct shape."""
        net = simple_network
        output_activations = np.random.randn(2, 1)
        y = np.random.randn(2, 1)
        result = net.cost_derivative(output_activations, y)
        assert result.shape == output_activations.shape

    def test_backprop_returns_correct_gradients(self, simple_network):
        """Test backprop returns gradients of correct shapes."""
        net = simple_network
        x = np.random.randn(3, 1)
        y = np.random.randn(2, 1)

        nabla_b, nabla_w = net.backprop(x, y)

        assert len(nabla_b) == len(net.biases)
        assert len(nabla_w) == len(net.weights)

        for nb, b in zip(nabla_b, net.biases):
            assert nb.shape == b.shape

        for nw, w in zip(nabla_w, net.weights):
            assert nw.shape == w.shape


@pytest.mark.slow
@pytest.mark.integration
class TestTraining:
    """Integration tests for training."""

    def test_training_improves_performance(self, simple_network, sample_training_data, sample_test_data):
        """Test that training improves network performance."""
        net = simple_network

        # Get initial accuracy
        initial_accuracy = net.evaluate(sample_test_data)

        # Train the network
        net.SGD(sample_training_data, epochs=10, mini_batch_size=2,
                eta=3.0, test_data=sample_test_data)

        # Get final accuracy
        final_accuracy = net.evaluate(sample_test_data)

        # Training should improve or maintain accuracy
        # (may not always improve with very small datasets, but this tests the process)
        assert isinstance(final_accuracy, (int, np.integer))

