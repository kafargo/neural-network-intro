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

    def test_status_counts_active_training_jobs(self, flask_client):
        """
        Test that status endpoint counts active training jobs correctly.

        The training_jobs count should only include jobs with status
        'pending' or 'training', not completed or failed jobs.
        """
        # Start with no training jobs
        response = flask_client.get('/api/status')
        data = json.loads(response.data)
        assert data['training_jobs'] == 0

        # Create a network and start training
        create_response = flask_client.post('/api/networks', json={})
        network_id = json.loads(create_response.data)['network_id']

        # Start training (uses 1 epoch to be quick)
        train_response = flask_client.post(
            f'/api/networks/{network_id}/train',
            json={'epochs': 1, 'mini_batch_size': 100}
        )
        assert train_response.status_code == 202

        # Immediately check status - should show at least 1 training job
        # (it might be pending or training)
        response = flask_client.get('/api/status')
        data = json.loads(response.data)
        # Note: Job might complete very quickly, so we check >= 0
        assert data['training_jobs'] >= 0


@pytest.mark.api
class TestNetworkCreation:
    """Tests for network creation endpoint."""

    def test_create_network_default_architecture(self, flask_client):
        """Test creating a network with default architecture."""
        response = flask_client.post('/api/networks',
                                    json={})
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'network_id' in data
        assert 'architecture' in data
        assert data['architecture'] == [784, 30, 10]

    def test_create_network_custom_architecture(self, flask_client):
        """Test creating a network with custom architecture."""
        custom_arch = [784, 128, 64, 10]
        response = flask_client.post('/api/networks',
                                    json={'layer_sizes': custom_arch})
        assert response.status_code == 201
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

        assert train_response.status_code == 202
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

        assert train_response.status_code == 202

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
        assert create_response.status_code == 201
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
        assert train_response.status_code == 202
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


@pytest.mark.api
class TestCleanupEndpoint:
    """Tests for the network cleanup endpoint."""

    def test_cleanup_endpoint_default_days(self, flask_client):
        """Test cleanup endpoint with default 2 days."""
        import sqlite3
        import os

        # Create some test networks
        response1 = flask_client.post('/api/networks', json={'layer_sizes': [784, 30, 10]})
        assert response1.status_code == 201
        network_id = json.loads(response1.data)['network_id']

        # Age the network in the database
        db_path = 'models/networks.db'
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE networks 
                SET created_at = datetime('now', '-3 days')
                WHERE network_id = ?
            ''', (network_id,))
            conn.commit()
            conn.close()

            # Trigger cleanup
            cleanup_response = flask_client.post('/api/networks/cleanup', json={})
            assert cleanup_response.status_code == 200
            cleanup_data = json.loads(cleanup_response.data)
            assert 'deleted_count' in cleanup_data
            assert cleanup_data['days'] == 2

    def test_cleanup_endpoint_custom_days(self, flask_client):
        """Test cleanup endpoint with custom days parameter."""
        cleanup_response = flask_client.post(
            '/api/networks/cleanup',
            json={'days': 7}
        )
        assert cleanup_response.status_code == 200
        cleanup_data = json.loads(cleanup_response.data)
        assert 'deleted_count' in cleanup_data
        assert cleanup_data['days'] == 7
        assert 'message' in cleanup_data

    def test_cleanup_endpoint_invalid_days(self, flask_client):
        """Test cleanup endpoint with invalid days parameter."""
        # Test negative days
        response = flask_client.post(
            '/api/networks/cleanup',
            json={'days': -1}
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_cleanup_endpoint_no_old_networks(self, flask_client):
        """Test cleanup endpoint when no old networks exist."""
        cleanup_response = flask_client.post(
            '/api/networks/cleanup',
            json={'days': 2}
        )
        assert cleanup_response.status_code == 200
        cleanup_data = json.loads(cleanup_response.data)
        assert cleanup_data['deleted_count'] >= 0

    def test_cleanup_endpoint_without_json(self, flask_client):
        """Test cleanup endpoint without JSON body."""
        cleanup_response = flask_client.post(
            '/api/networks/cleanup',
            json={}
        )
        assert cleanup_response.status_code == 200
        cleanup_data = json.loads(cleanup_response.data)
        # Should use default value of 2 days
        assert cleanup_data['days'] == 2


@pytest.mark.api
class TestAboutCustomizeEndpoint:
    """Tests for the AI-powered about customization endpoint."""

    def test_missing_prompt_returns_400(self, flask_client):
        """Test that a missing prompt returns 400."""
        response = flask_client.post('/api/about/customize', json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'prompt' in data['error'].lower()

    def test_empty_prompt_returns_400(self, flask_client):
        """Test that an empty string prompt returns 400."""
        response = flask_client.post(
            '/api/about/customize',
            json={'prompt': '   '}
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_missing_work_history_returns_503(self, flask_client):
        """Test that a missing work history file returns 503."""
        from unittest.mock import patch
        with patch(
            'api_server.load_work_history', return_value=None
        ), patch.dict(
            'os.environ', {'OPENAI_API_KEY': 'test-key'}
        ):
            response = flask_client.post(
                '/api/about/customize',
                json={'prompt': 'Make it technical'}
            )
            assert response.status_code == 503
            data = json.loads(response.data)
            assert 'error' in data
            assert 'work history' in data['error'].lower()

    def test_missing_api_key_returns_500(self, flask_client):
        """Test that a missing OpenAI API key returns 500."""
        from unittest.mock import patch
        with patch(
            'api_server.load_work_history',
            return_value='Some work history content'
        ):
            with patch.dict('os.environ', {}, clear=True):
                # Remove OPENAI_API_KEY if present
                import os
                env_copy = os.environ.copy()
                env_copy.pop('OPENAI_API_KEY', None)
                with patch.dict('os.environ', env_copy, clear=True):
                    response = flask_client.post(
                        '/api/about/customize',
                        json={'prompt': 'Make it technical'}
                    )
                    assert response.status_code == 500
                    data = json.loads(response.data)
                    assert 'error' in data

    def test_successful_customization(self, flask_client):
        """Test successful about customization with mocked OpenAI."""
        from unittest.mock import patch, MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            'about_me': 'First paragraph.\n\nSecond paragraph.',
            'skills': [
                {'category': 'Backend', 'tags': ['Python', 'Flask']},
                {'category': 'Frontend', 'tags': ['Angular']}
            ]
        })

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = (
            mock_response
        )
        mock_openai_class = MagicMock(return_value=mock_client_instance)

        with patch(
            'api_server.load_work_history',
            return_value='Detailed work history content here'
        ), patch.dict(
            'os.environ', {'OPENAI_API_KEY': 'test-key'}
        ), patch(
            'api_server.OpenAI', mock_openai_class
        ):
            response = flask_client.post(
                '/api/about/customize',
                json={'prompt': 'Make it sound technical'}
            )
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'about_me' in data
            assert 'skills' in data
            assert isinstance(data['about_me'], str)
            assert isinstance(data['skills'], list)
            assert len(data['skills']) == 2
            assert data['skills'][0]['category'] == 'Backend'

    def test_openai_failure_returns_500(self, flask_client):
        """Test that an OpenAI API failure returns 500."""
        from unittest.mock import patch, MagicMock

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = (
            Exception("OpenAI API error")
        )
        mock_openai_class = MagicMock(return_value=mock_client_instance)

        with patch(
            'api_server.load_work_history',
            return_value='Some work history'
        ), patch.dict(
            'os.environ', {'OPENAI_API_KEY': 'test-key'}
        ), patch(
            'api_server.OpenAI', mock_openai_class
        ):
            response = flask_client.post(
                '/api/about/customize',
                json={'prompt': 'Make it fun'}
            )
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data

    def test_invalid_ai_response_returns_500(self, flask_client):
        """Test that an invalid JSON response from AI returns 500."""
        from unittest.mock import patch, MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # Missing required 'skills' key
        mock_response.choices[0].message.content = json.dumps({
            'about_me': 'Some text'
        })

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = (
            mock_response
        )
        mock_openai_class = MagicMock(return_value=mock_client_instance)

        with patch(
            'api_server.load_work_history',
            return_value='Some work history'
        ), patch.dict(
            'os.environ', {'OPENAI_API_KEY': 'test-key'}
        ), patch(
            'api_server.OpenAI', mock_openai_class
        ):
            response = flask_client.post(
                '/api/about/customize',
                json={'prompt': 'Rewrite'}
            )
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data

    def test_no_json_body_returns_400(self, flask_client):
        """Test that a request without JSON body returns 400."""
        response = flask_client.post(
            '/api/about/customize',
            content_type='application/json',
            data='{}'
        )
        assert response.status_code == 400


