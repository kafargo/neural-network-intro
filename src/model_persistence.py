"""
model_persistence.py
~~~~~~~~~~~~~~~~~

A module to handle the persistence of neural network models
to allow saving and loading models for later use.
"""

import os
import pickle
import json
import numpy as np

class NetworkEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_network(network, network_id, model_dir='models', trained=True, accuracy=None):
    """
    Save a trained network to disk
    
    Args:
        network: The neural network object to save
        network_id: A unique identifier for the network
        model_dir: Directory to save the model in
        trained: Boolean indicating if the network has been trained
        accuracy: The accuracy of the trained network (0.0 to 1.0)

    Returns:
        bool: True if the save was successful
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Save the network object
    with open(f"{model_dir}/{network_id}.pkl", 'wb') as f:
        pickle.dump(network, f)
    
    # Save network metadata including trained status and accuracy
    metadata = {
        'network_id': network_id,
        'architecture': network.sizes,
        'weights_shape': [w.shape for w in network.weights],
        'biases_shape': [b.shape for b in network.biases],
        'trained': trained,
        'accuracy': accuracy
    }
    
    with open(f"{model_dir}/{network_id}.json", 'w') as f:
        json.dump(metadata, f, cls=NetworkEncoder)
    
    return True

def load_network(network_id, model_dir='models'):
    """
    Load a trained network from disk
    
    Args:
        network_id: The unique identifier of the network to load
        model_dir: Directory where the model is stored
        
    Returns:
        The loaded neural network object or None if not found
    """
    network_path = f"{model_dir}/{network_id}.pkl"
    
    if not os.path.exists(network_path):
        return None
    
    with open(network_path, 'rb') as f:
        network = pickle.load(f)
    
    return network

def list_saved_networks(model_dir='models'):
    """
    List all saved networks
    
    Args:
        model_dir: Directory where models are stored
        
    Returns:
        list: A list of metadata for each saved network
    """
    if not os.path.exists(model_dir):
        return []
    
    networks = []
    for filename in os.listdir(model_dir):
        if filename.endswith('.json'):
            network_id = filename.split('.')[0]
            with open(f"{model_dir}/{filename}", 'r') as f:
                metadata = json.load(f)

            # Ensure backwards compatibility: add trained and accuracy if not present
            if 'trained' not in metadata:
                # If a network was saved, it was trained (old behavior)
                metadata['trained'] = True
            if 'accuracy' not in metadata:
                metadata['accuracy'] = None

            networks.append(metadata)
    
    return networks

def delete_network(network_id, model_dir='models'):
    """
    Delete a saved network
    
    Args:
        network_id: The unique identifier of the network to delete
        model_dir: Directory where the model is stored
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    network_path = f"{model_dir}/{network_id}.pkl"
    metadata_path = f"{model_dir}/{network_id}.json"
    
    if not os.path.exists(network_path) and not os.path.exists(metadata_path):
        return False
    
    try:
        if os.path.exists(network_path):
            os.remove(network_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        return True
    except Exception as e:
        print(f"Error deleting network: {e}")
        return False
