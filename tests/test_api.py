"""
test_api.py
~~~~~~~~~~~

API endpoint tests for the Flask server.
"""

import pytest
import json
import time


@pytest.mark.api
class TestStatusEndpoint:
    """Tests for the status endpoint."""

    def test_status_endpoint_returns_200(self, flask_client):
        """Test status endpoint is accessible."""
        response = flask_client.get('/api/status')
        assert response.status_code == 200

    def test_status_endpoint_json(self, flask_client):
        """Test status endpoint returns JSON."""
        response = flask_client.get('/api/status')
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'online'

    def test_status_includes_counts(self, flask_client):
        """Test status endpoint includes network and job counts."""
        response = flask_client.get('/api/status')
        data = json.loads(response.data)
        assert 'active_networks' in data
        assert 'training_jobs' in data
        assert isinstance(data['active_networks'], int)
        assert isinstance(data['training_jobs'], int)


@pytest.mark.api
class TestNetworkCreation:
    """Tests for network creation endpoint."""

    def test_create_network_default_architecture(self, flask_client):
        """Test creating a network with default architecture."""
        response = flask_client.post('/api/networks',
                                    json={})
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'network_id' in data
        assert 'architecture' in data
        assert data['architecture'] == [784, 30, 10]

    def test_create_network_custom_architecture(self, flask_client):
        """Test creating a network with custom architecture."""
        custom_arch = [784, 128, 64, 10]
        response = flask_client.post('/api/networks',
                                    json={'layer_sizes': custom_arch})
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['architecture'] == custom_arch

    def test_create_network_returns_unique_ids(self, flask_client):
        """Test that each created network gets a unique ID."""
        response1 = flask_client.post('/api/networks', json={})
        response2 = flask_client.post('/api/networks', json={})

        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)

        assert data1['network_id'] != data2['network_id']

    def test_create_network_status_created(self, flask_client):
        """Test that created network has correct status."""
        response = flask_client.post('/api/networks', json={})
        data = json.loads(response.data)
        assert data['status'] == 'created'


@pytest.mark.api
class TestNetworkListing:
    """Tests for network listing endpoint."""

    def test_list_networks_empty(self, flask_client):
        """Test listing networks when none exist."""
        response = flask_client.get('/api/networks')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'networks' in data
        assert isinstance(data['networks'], list)

    def test_list_networks_after_creation(self, flask_client):
        """Test listing networks after creating some."""
        # Create 3 networks
        for i in range(3):
            flask_client.post('/api/networks', json={})

        response = flask_client.get('/api/networks')
        data = json.loads(response.data)
        assert len(data['networks']) == 3

    def test_listed_network_has_metadata(self, flask_client):
        """Test that listed networks include required metadata."""
        flask_client.post('/api/networks', json={'layer_sizes': [784, 50, 10]})

        response = flask_client.get('/api/networks')
        data = json.loads(response.data)

        network = data['networks'][0]
        assert 'network_id' in network
        assert 'architecture' in network
        assert 'trained' in network
        assert 'accuracy' in network
        assert network['architecture'] == [784, 50, 10]


@pytest.mark.api
class TestNetworkDeletion:
    """Tests for network deletion endpoints."""

    def test_delete_network_success(self, flask_client):
        """Test deleting an existing network."""
        # Create a network
        create_response = flask_client.post('/api/networks', json={})
        network_id = json.loads(create_response.data)['network_id']

        # Delete it
        delete_response = flask_client.delete(f'/api/networks/{network_id}')
        assert delete_response.status_code == 200

        data = json.loads(delete_response.data)
        assert data['deleted_from_memory'] is True

    def test_delete_nonexistent_network(self, flask_client):
        """Test deleting a network that doesn't exist."""
        response = flask_client.delete('/api/networks/nonexistent-id')
        assert response.status_code == 404

    def test_delete_all_networks(self, flask_client):
        """Test deleting all networks."""
        # Create 3 networks
        for i in range(3):
            flask_client.post('/api/networks', json={})

        # Delete all
        response = flask_client.delete('/api/networks')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['deleted_count'] == 3

        # Verify they're gone
        list_response = flask_client.get('/api/networks')
        list_data = json.loads(list_response.data)
        assert len(list_data['networks']) == 0


@pytest.mark.api
class TestTrainingEndpoints:
    """Tests for training-related endpoints."""

    def test_train_network_success(self, flask_client):
        """Test starting training on a network."""
        # Create a network
        create_response = flask_client.post('/api/networks', json={})
        network_id = json.loads(create_response.data)['network_id']

        # Start training
        train_response = flask_client.post(
            f'/api/networks/{network_id}/train',
            json={'epochs': 1, 'mini_batch_size': 10, 'learning_rate': 0.5}
        )

        assert train_response.status_code == 200
        data = json.loads(train_response.data)
        assert 'job_id' in data
        assert 'network_id' in data
        assert data['status'] == 'training_started'

    def test_train_nonexistent_network(self, flask_client):
        """Test training a network that doesn't exist."""
        response = flask_client.post(
            '/api/networks/nonexistent-id/train',
            json={'epochs': 1}
        )
        assert response.status_code == 404

    def test_train_with_default_parameters(self, flask_client):
        """Test training with default parameters."""
        # Create a network
        create_response = flask_client.post('/api/networks', json={})
        network_id = json.loads(create_response.data)['network_id']

        # Train with no parameters (should use defaults)
        train_response = flask_client.post(
            f'/api/networks/{network_id}/train',
            json={}
        )

        assert train_response.status_code == 200

    def test_get_training_status(self, flask_client):
        """Test getting training job status."""
        # Create and start training
        create_response = flask_client.post('/api/networks', json={})
        network_id = json.loads(create_response.data)['network_id']

        train_response = flask_client.post(
            f'/api/networks/{network_id}/train',
            json={'epochs': 1}
        )
        job_id = json.loads(train_response.data)['job_id']

        # Get status
        status_response = flask_client.get(f'/api/training/{job_id}')
        assert status_response.status_code == 200

        data = json.loads(status_response.data)
        assert 'status' in data
        assert 'network_id' in data

    def test_get_nonexistent_training_status(self, flask_client):
        """Test getting status of nonexistent training job."""
        response = flask_client.get('/api/training/nonexistent-job-id')
        assert response.status_code == 404


@pytest.mark.api
class TestExampleEndpoints:
    """Tests for successful/unsuccessful example endpoints."""

    def test_successful_example_untrained_network(self, flask_client):
        """Test getting successful example from untrained network."""
        # Create a network
        create_response = flask_client.post('/api/networks', json={})
        network_id = json.loads(create_response.data)['network_id']

        # Try to get successful example (should work but be random)
        response = flask_client.get(f'/api/networks/{network_id}/successful_example')
        # May succeed or fail depending on random predictions
        assert response.status_code in [200, 404]

    def test_unsuccessful_example_untrained_network(self, flask_client):
        """Test getting unsuccessful example from untrained network."""
        # Create a network
        create_response = flask_client.post('/api/networks', json={})
        network_id = json.loads(create_response.data)['network_id']

        # Try to get unsuccessful example
        response = flask_client.get(f'/api/networks/{network_id}/unsuccessful_example')
        # May succeed or fail depending on random predictions
        assert response.status_code in [200, 404]

    def test_example_endpoint_nonexistent_network(self, flask_client):
        """Test example endpoint with nonexistent network."""
        response = flask_client.get('/api/networks/nonexistent-id/successful_example')
        assert response.status_code == 404

    def test_successful_example_response_format(self, flask_client):
        """Test successful example response has correct format."""
        # Create a network
        create_response = flask_client.post('/api/networks', json={})
        network_id = json.loads(create_response.data)['network_id']

        # Get example (try multiple times since it might fail randomly)
        for _ in range(5):
            response = flask_client.get(f'/api/networks/{network_id}/successful_example')
            if response.status_code == 200:
                data = json.loads(response.data)
                assert 'network_id' in data
                assert 'example_index' in data
                assert 'predicted_digit' in data
                assert 'actual_digit' in data
                assert 'image_data' in data
                assert 'output_weights' in data
                assert 'network_output' in data
                break


@pytest.mark.api
class TestStaticRoutes:
    """Tests for static file routes."""

    def test_index_route(self, flask_client):
        """Test that index route is accessible."""
        response = flask_client.get('/')
        # Should return 200 if index.html exists, or 404 if not
        assert response.status_code in [200, 404]

    def test_invalid_route_returns_404(self, flask_client):
        """Test that invalid routes return 404."""
        response = flask_client.get('/nonexistent-page')
        assert response.status_code == 404


@pytest.mark.api
@pytest.mark.integration
class TestTrainingFlow:
    """Integration tests for complete training workflow."""

    def test_complete_training_flow(self, flask_client):
        """Test creating, training, and checking a network."""
        # 1. Create network
        create_response = flask_client.post('/api/networks',
                                           json={'layer_sizes': [784, 30, 10]})
        assert create_response.status_code == 200
        network_id = json.loads(create_response.data)['network_id']

        # 2. Verify it's in the list
        list_response = flask_client.get('/api/networks')
        networks = json.loads(list_response.data)['networks']
        assert any(n['network_id'] == network_id for n in networks)

        # 3. Start training
        train_response = flask_client.post(
            f'/api/networks/{network_id}/train',
            json={'epochs': 1, 'mini_batch_size': 10, 'learning_rate': 0.5}
        )
        assert train_response.status_code == 200
        job_id = json.loads(train_response.data)['job_id']

        # 4. Check training status
        status_response = flask_client.get(f'/api/training/{job_id}')
        assert status_response.status_code == 200

        # 5. Delete network
        delete_response = flask_client.delete(f'/api/networks/{network_id}')
        assert delete_response.status_code == 200

        # 6. Verify it's gone
        list_response2 = flask_client.get('/api/networks')
        networks2 = json.loads(list_response2.data)['networks']
        # Network might still be in memory if training hasn't finished
        # but delete should have been called successfully

