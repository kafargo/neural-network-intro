"""
test_model_persistence.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the model persistence module.
"""

import pytest
import os
import json
from network import Network
from model_persistence import (
    save_network,
    load_network,
    list_saved_networks,
    delete_network
)


@pytest.mark.unit
class TestModelPersistence:
    """Tests for model persistence functions."""

    def test_save_network_creates_files(self, simple_network, temp_model_dir):
        """Test that saving a network creates the expected files."""
        network_id = "test_network_001"
        result = save_network(simple_network, network_id, model_dir=temp_model_dir)

        assert result is True
        assert os.path.exists(f"{temp_model_dir}/{network_id}.pkl")
        assert os.path.exists(f"{temp_model_dir}/{network_id}.json")

    def test_save_network_metadata(self, simple_network, temp_model_dir):
        """Test that saved metadata contains correct information."""
        network_id = "test_network_002"
        accuracy = 0.95
        save_network(simple_network, network_id, model_dir=temp_model_dir,
                    trained=True, accuracy=accuracy)

        with open(f"{temp_model_dir}/{network_id}.json", 'r') as f:
            metadata = json.load(f)

        assert metadata['network_id'] == network_id
        assert metadata['architecture'] == simple_network.sizes
        assert metadata['trained'] is True
        assert metadata['accuracy'] == accuracy

    def test_load_network_returns_network(self, simple_network, temp_model_dir):
        """Test that loading a network returns a Network object."""
        network_id = "test_network_003"
        save_network(simple_network, network_id, model_dir=temp_model_dir)

        loaded_network = load_network(network_id, model_dir=temp_model_dir)

        assert loaded_network is not None
        assert isinstance(loaded_network, Network)
        assert loaded_network.sizes == simple_network.sizes

    def test_load_nonexistent_network(self, temp_model_dir):
        """Test that loading a nonexistent network returns None."""
        result = load_network("nonexistent_network", model_dir=temp_model_dir)
        assert result is None

    def test_load_preserves_weights(self, simple_network, temp_model_dir):
        """Test that loaded network has same weights as saved network."""
        network_id = "test_network_004"

        # Save original weights
        original_weights = [w.copy() for w in simple_network.weights]

        save_network(simple_network, network_id, model_dir=temp_model_dir)
        loaded_network = load_network(network_id, model_dir=temp_model_dir)

        # Compare weights
        for orig, loaded in zip(original_weights, loaded_network.weights):
            assert orig.shape == loaded.shape
            assert (orig == loaded).all()

    def test_list_saved_networks_empty(self, temp_model_dir):
        """Test listing networks in empty directory."""
        networks = list_saved_networks(model_dir=temp_model_dir)
        assert networks == []

    def test_list_saved_networks(self, simple_network, temp_model_dir):
        """Test listing saved networks."""
        # Save multiple networks
        network_ids = ["net_001", "net_002", "net_003"]
        for net_id in network_ids:
            save_network(simple_network, net_id, model_dir=temp_model_dir,
                        trained=True, accuracy=0.9)

        networks = list_saved_networks(model_dir=temp_model_dir)

        assert len(networks) == 3
        returned_ids = [net['network_id'] for net in networks]
        for net_id in network_ids:
            assert net_id in returned_ids

    def test_list_saved_networks_includes_metadata(self, simple_network, temp_model_dir):
        """Test that listed networks include metadata."""
        network_id = "net_with_metadata"
        accuracy = 0.87
        save_network(simple_network, network_id, model_dir=temp_model_dir,
                    trained=True, accuracy=accuracy)

        networks = list_saved_networks(model_dir=temp_model_dir)

        assert len(networks) == 1
        net = networks[0]
        assert net['network_id'] == network_id
        assert net['architecture'] == simple_network.sizes
        assert net['trained'] is True
        assert net['accuracy'] == accuracy

    def test_delete_network_removes_files(self, simple_network, temp_model_dir):
        """Test that deleting a network removes its files."""
        network_id = "net_to_delete"
        save_network(simple_network, network_id, model_dir=temp_model_dir)

        # Verify files exist
        assert os.path.exists(f"{temp_model_dir}/{network_id}.pkl")
        assert os.path.exists(f"{temp_model_dir}/{network_id}.json")

        # Delete network
        result = delete_network(network_id, model_dir=temp_model_dir)

        assert result is True
        assert not os.path.exists(f"{temp_model_dir}/{network_id}.pkl")
        assert not os.path.exists(f"{temp_model_dir}/{network_id}.json")

    def test_delete_nonexistent_network(self, temp_model_dir):
        """Test that deleting a nonexistent network returns False."""
        result = delete_network("nonexistent", model_dir=temp_model_dir)
        assert result is False

    def test_save_untrained_network(self, simple_network, temp_model_dir):
        """Test saving an untrained network."""
        network_id = "untrained_net"
        save_network(simple_network, network_id, model_dir=temp_model_dir,
                    trained=False, accuracy=None)

        with open(f"{temp_model_dir}/{network_id}.json", 'r') as f:
            metadata = json.load(f)

        assert metadata['trained'] is False
        assert metadata['accuracy'] is None


@pytest.mark.integration
class TestPersistenceIntegration:
    """Integration tests for model persistence."""

    def test_save_load_train_cycle(self, simple_network, sample_training_data, temp_model_dir):
        """Test full cycle: save, load, and train."""
        network_id = "cycle_test"

        # Save initial network
        save_network(simple_network, network_id, model_dir=temp_model_dir, trained=False)

        # Load it back
        loaded_network = load_network(network_id, model_dir=temp_model_dir)

        # Train the loaded network
        loaded_network.SGD(sample_training_data, epochs=2,
                          mini_batch_size=2, eta=0.5)

        # Save trained version
        save_network(loaded_network, network_id, model_dir=temp_model_dir,
                    trained=True, accuracy=0.8)

        # Verify it worked
        networks = list_saved_networks(model_dir=temp_model_dir)
        assert len(networks) == 1
        assert networks[0]['trained'] is True

    def test_multiple_networks_coexist(self, mnist_network, simple_network, temp_model_dir):
        """Test that multiple different networks can be saved and loaded."""
        save_network(mnist_network, "mnist_net", model_dir=temp_model_dir)
        save_network(simple_network, "simple_net", model_dir=temp_model_dir)

        networks = list_saved_networks(model_dir=temp_model_dir)
        assert len(networks) == 2

        # Load both and verify they're different
        loaded_mnist = load_network("mnist_net", model_dir=temp_model_dir)
        loaded_simple = load_network("simple_net", model_dir=temp_model_dir)

        assert loaded_mnist.sizes == [784, 30, 10]
        assert loaded_simple.sizes == [3, 4, 2]

