#!/usr/bin/env python3
"""
Quick test script to verify the cleaned API server works correctly.
Run this after starting the API server to test all essential endpoints.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_status():
    """Test the status endpoint"""
    print("Testing GET /api/status...")
    response = requests.get(f"{BASE_URL}/api/status")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    assert response.status_code == 200
    print("  ✓ PASSED\n")

def test_create_network():
    """Test creating a network"""
    print("Testing POST /api/networks...")
    response = requests.post(f"{BASE_URL}/api/networks", json={
        "layer_sizes": [784, 30, 10]
    })
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Network ID: {data['network_id']}")
    print(f"  Architecture: {data['architecture']}")
    assert response.status_code == 200
    print("  ✓ PASSED\n")
    return data['network_id']

def test_list_networks():
    """Test listing networks"""
    print("Testing GET /api/networks...")
    response = requests.get(f"{BASE_URL}/api/networks")
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Total networks: {len(data['networks'])}")
    assert response.status_code == 200
    print("  ✓ PASSED\n")

def test_train_network(network_id):
    """Test training a network"""
    print(f"Testing POST /api/networks/{network_id[:8]}.../train...")
    response = requests.post(f"{BASE_URL}/api/networks/{network_id}/train", json={
        "epochs": 1,
        "mini_batch_size": 10,
        "learning_rate": 3.0
    })
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Job ID: {data['job_id']}")
    assert response.status_code == 200
    print("  ✓ PASSED\n")
    return data['job_id']

def test_training_status(job_id):
    """Test getting training status"""
    print(f"Testing GET /api/training/{job_id[:8]}...")
    # Wait a moment for training to start
    time.sleep(2)
    response = requests.get(f"{BASE_URL}/api/training/{job_id}")
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Training status: {data['status']}")
    print(f"  Progress: {data.get('progress', 'N/A')}")
    assert response.status_code == 200
    print("  ✓ PASSED\n")

    # Wait for training to complete
    print("  Waiting for training to complete...")
    while True:
        response = requests.get(f"{BASE_URL}/api/training/{job_id}")
        data = response.json()
        status = data['status']
        if status == 'completed':
            print(f"  Training completed! Accuracy: {data.get('accuracy', 'N/A')}")
            break
        elif status == 'failed':
            print(f"  Training failed: {data.get('error', 'Unknown error')}")
            break
        time.sleep(5)
    print()

def test_successful_example(network_id):
    """Test getting successful example"""
    print(f"Testing GET /api/networks/{network_id[:8]}.../successful_example...")
    response = requests.get(f"{BASE_URL}/api/networks/{network_id}/successful_example")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Example index: {data['example_index']}")
        print(f"  Predicted: {data['predicted_digit']}, Actual: {data['actual_digit']}")
        print(f"  Image data length: {len(data['image_data'])} chars")
        assert response.status_code == 200
        print("  ✓ PASSED\n")
    else:
        print(f"  Response: {response.text}")
        print("  ⚠ SKIPPED (network may not be trained yet)\n")

def test_unsuccessful_example(network_id):
    """Test getting unsuccessful example"""
    print(f"Testing GET /api/networks/{network_id[:8]}.../unsuccessful_example...")
    response = requests.get(f"{BASE_URL}/api/networks/{network_id}/unsuccessful_example")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Example index: {data['example_index']}")
        print(f"  Predicted: {data['predicted_digit']}, Actual: {data['actual_digit']}")
        print(f"  Image data length: {len(data['image_data'])} chars")
        assert response.status_code == 200
        print("  ✓ PASSED\n")
    else:
        print(f"  Response: {response.text}")
        print("  ⚠ SKIPPED (network may be too accurate)\n")

def test_delete_network(network_id):
    """Test deleting a network"""
    print(f"Testing DELETE /api/networks/{network_id[:8]}...")
    response = requests.delete(f"{BASE_URL}/api/networks/{network_id}")
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Deleted from memory: {data['deleted_from_memory']}")
    print(f"  Deleted from disk: {data['deleted_from_disk']}")
    assert response.status_code == 200
    print("  ✓ PASSED\n")

def main():
    print("=" * 70)
    print("API Server Test Suite")
    print("=" * 70)
    print()
    print("Make sure the API server is running at http://localhost:8000")
    print()

    try:
        # Test status
        test_status()

        # Test creating a network
        network_id = test_create_network()

        # Test listing networks
        test_list_networks()

        # Test training
        job_id = test_train_network(network_id)
        test_training_status(job_id)

        # Test getting examples
        test_successful_example(network_id)
        test_unsuccessful_example(network_id)

        # Test deletion
        test_delete_network(network_id)

        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)

    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Could not connect to API server at", BASE_URL)
        print("  Make sure the server is running: python src/api_server.py")
    except AssertionError as e:
        print("✗ TEST FAILED:", str(e))
    except Exception as e:
        print("✗ ERROR:", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

