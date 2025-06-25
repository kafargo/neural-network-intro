"""
api_server.py
~~~~~~~~~~~~~~~~~

A Flask-based API server that serves as the backend for the neural network
application. It enables frontend applications to create networks, train them,
and retrieve results.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import json
import base64
from io import BytesIO
import threading
# Set matplotlib to use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a display
import matplotlib.pyplot as plt
import numpy as np

# Import our existing network code
try:
    # Try direct imports first (when running as a module)
    from src import network
    from src import mnist_loader
    from src.visualization_helpers import (
        visualize_network_structure,
        run_specific_examples,
        find_misclassified_examples,
        display_digit
    )
    from src.model_persistence import (
        save_network,
        load_network,
        list_saved_networks,
        delete_network
    )
except ImportError:
    # Fall back to relative imports (when running directly)
    import network
    import mnist_loader
    from visualization_helpers import (
        visualize_network_structure,
        run_specific_examples,
        find_misclassified_examples,
        display_digit
    )
    from model_persistence import (
        save_network,
        load_network,
        list_saved_networks,
        delete_network
    )

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and all origins

# Store active networks in memory
active_networks = {}

# Store training jobs
training_jobs = {}

# Load MNIST data once at startup
print("Loading MNIST data...")
try:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

@app.route('/api/status', methods=['GET'])
def get_status():
    """Basic endpoint to check if the API is running"""
    return jsonify({
        'status': 'online',
        'active_networks': len(active_networks),
        'training_jobs': len(training_jobs)
    })

@app.route('/api/networks', methods=['POST'])
def create_network():
    """Create a new neural network with the specified architecture"""
    data = request.get_json()
    
    # Get layer sizes from request, default to [784, 30, 10] if not specified
    layer_sizes = data.get('layer_sizes', [784, 30, 10])
    
    # Create a unique ID for this network
    network_id = str(uuid.uuid4())
    
    # Create the network with specified architecture
    net = network.Network(layer_sizes)
    
    # Store in our dictionary
    active_networks[network_id] = {
        'network': net,
        'architecture': layer_sizes,
        'trained': False,
        'accuracy': None
    }
    
    return jsonify({
        'network_id': network_id,
        'architecture': layer_sizes,
        'status': 'created'
    })

@app.route('/api/networks/<network_id>/train', methods=['POST'])
def train_network(network_id):
    """Start asynchronous training for the specified network"""
    if network_id not in active_networks:
        return jsonify({'error': 'Network not found'}), 404
        
    data = request.get_json()
    epochs = data.get('epochs', 5)
    mini_batch_size = data.get('mini_batch_size', 10)
    learning_rate = data.get('learning_rate', 3.0)
    
    # Create a job ID for this training task
    job_id = str(uuid.uuid4())
    
    # Set up the training job status
    training_jobs[job_id] = {
        'network_id': network_id,
        'status': 'pending',
        'progress': 0,
        'epochs': epochs
    }
    
    # Start training in a separate thread
    thread = threading.Thread(
        target=train_network_task,
        args=(network_id, job_id, epochs, mini_batch_size, learning_rate)
    )
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'network_id': network_id,
        'status': 'training_started'
    })

def train_network_task(network_id, job_id, epochs, mini_batch_size, learning_rate):
    """Background task to train the network"""
    net = active_networks[network_id]['network']
    
    try:
        # Train the network
        net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)
        
        # Calculate accuracy
        accuracy = net.evaluate(test_data) / len(test_data)
        
        # Update network status
        active_networks[network_id]['trained'] = True
        active_networks[network_id]['accuracy'] = accuracy
        
        # Update job status
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['accuracy'] = accuracy
        
        # Save the trained network
        save_network(net, network_id)
        
    except Exception as e:
        # Update job status on error
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)

@app.route('/api/training/<job_id>', methods=['GET'])
def get_training_status(job_id):
    """Get the status of a training job"""
    if job_id not in training_jobs:
        return jsonify({'error': 'Training job not found'}), 404
    
    return jsonify(training_jobs[job_id])

@app.route('/api/networks/<network_id>/predict', methods=['POST'])
def predict(network_id):
    """Run prediction on a specific example"""
    if network_id not in active_networks:
        return jsonify({'error': 'Network not found'}), 404
    
    data = request.get_json()
    example_index = data.get('example_index', 0)
    
    if example_index >= len(test_data):
        return jsonify({'error': 'Example index out of range'}), 400
        
    net = active_networks[network_id]['network']
    
    x, y = test_data[example_index]
    output = net.feedforward(x)
    predicted_digit = int(np.argmax(output))
    actual_digit = int(y)
    
    # Convert numpy arrays to list for JSON serialization
    output_list = [float(val) for val in output]
    
    return jsonify({
        'example_index': example_index,
        'predicted_digit': predicted_digit,
        'actual_digit': actual_digit,
        'confidence_scores': output_list,
        'correct': predicted_digit == actual_digit
    })

@app.route('/api/networks/<network_id>/predict_batch', methods=['POST'])
def predict_batch(network_id):
    """Run prediction on a batch of examples"""
    if network_id not in active_networks:
        return jsonify({'error': 'Network not found'}), 404
    
    data = request.get_json()
    start_index = data.get('start_index', 0)
    count = data.get('count', 10)
    
    if start_index >= len(test_data):
        return jsonify({'error': 'Start index out of range'}), 400
    
    end_index = min(start_index + count, len(test_data))
    
    net = active_networks[network_id]['network']
    
    results = []
    for i in range(start_index, end_index):
        x, y = test_data[i]
        output = net.feedforward(x)
        predicted_digit = int(np.argmax(output))
        actual_digit = int(y)
        
        # Convert image data to base64 for sending to frontend
        plt.figure(figsize=(2, 2))
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        results.append({
            'example_index': i,
            'predicted_digit': predicted_digit,
            'actual_digit': actual_digit,
            'correct': predicted_digit == actual_digit,
            'image_data': img_base64
        })
    
    return jsonify({
        'results': results,
        'total': end_index - start_index
    })

@app.route('/api/networks/<network_id>/visualize', methods=['GET'])
def get_visualization(network_id):
    """Return a visualization of the network structure"""
    if network_id not in active_networks:
        return jsonify({'error': 'Network not found'}), 404
        
    net = active_networks[network_id]['network']
    
    # Create visualization in memory
    plt.figure(figsize=(12, 8))
    visualize_network_structure(net, display=False)
    
    # Save to in-memory bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Encode as base64 for easy frontend display
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return jsonify({
        'network_id': network_id,
        'visualization': img_base64
    })

@app.route('/api/networks/<network_id>/misclassified', methods=['GET'])
def get_misclassified(network_id):
    """Find misclassified examples"""
    if network_id not in active_networks:
        return jsonify({'error': 'Network not found'}), 404
    
    max_count = request.args.get('max_count', default=10, type=int)
    max_check = request.args.get('max_check', default=500, type=int)
        
    net = active_networks[network_id]['network']
    
    # Find misclassified examples
    misclassified = find_misclassified_examples(net, test_data, max_count=max_count, max_check=max_check)
    
    results = []
    for idx in misclassified:
        x, y = test_data[idx]
        output = net.feedforward(x)
        predicted_digit = int(np.argmax(output))
        actual_digit = int(y)
        
        # Convert image data to base64 for sending to frontend
        plt.figure(figsize=(2, 2))
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        results.append({
            'example_index': idx,
            'predicted_digit': predicted_digit,
            'actual_digit': actual_digit,
            'image_data': img_base64
        })
    
    return jsonify({
        'misclassified': results
    })

@app.route('/api/networks', methods=['GET'])
def list_networks():
    """List all available networks"""
    # Combine in-memory and saved networks
    in_memory = [
        {
            'network_id': nid,
            'architecture': info['architecture'],
            'trained': info['trained'],
            'accuracy': info['accuracy'],
            'status': 'in_memory'
        }
        for nid, info in active_networks.items()
    ]
    
    saved = list_saved_networks()
    for net in saved:
        net['status'] = 'saved'
    
    return jsonify({
        'networks': in_memory + saved
    })

@app.route('/api/networks/<network_id>/load', methods=['POST'])
def load_saved_network(network_id):
    """Load a saved network into memory"""
    # Check if already in memory
    if network_id in active_networks:
        return jsonify({
            'network_id': network_id,
            'status': 'already_loaded'
        })
    
    # Try to load from disk
    net = load_network(network_id)
    if net is None:
        return jsonify({'error': 'Saved network not found'}), 404
    
    # Add to active networks
    active_networks[network_id] = {
        'network': net,
        'architecture': net.sizes,
        'trained': True,
        'accuracy': None  # Will need to evaluate to get this
    }
    
    return jsonify({
        'network_id': network_id,
        'architecture': net.sizes,
        'status': 'loaded'
    })

@app.route('/api/networks/<network_id>', methods=['DELETE'])
def delete_network_endpoint(network_id):
    """Delete a network (both from memory and disk)"""
    # Remove from active networks if present
    deleted_from_memory = False
    if network_id in active_networks:
        del active_networks[network_id]
        deleted_from_memory = True
    
    # Delete from disk if present
    deleted_from_disk = delete_network(network_id)
    
    if not deleted_from_memory and not deleted_from_disk:
        return jsonify({'error': 'Network not found'}), 404
    
    return jsonify({
        'network_id': network_id,
        'deleted_from_memory': deleted_from_memory,
        'deleted_from_disk': deleted_from_disk
    })

@app.route('/api/networks/<network_id>/stats', methods=['GET'])
def get_network_stats(network_id):
    """Get statistical information about a network"""
    if network_id not in active_networks:
        return jsonify({'error': 'Network not found'}), 404
        
    net = active_networks[network_id]['network']
    
    # Evaluate network on test data if not already done
    if active_networks[network_id]['accuracy'] is None and active_networks[network_id]['trained']:
        accuracy = net.evaluate(test_data) / len(test_data)
        active_networks[network_id]['accuracy'] = accuracy
    else:
        accuracy = active_networks[network_id]['accuracy']
    
    # Get weight and bias statistics
    weight_stats = []
    for i, w in enumerate(net.weights):
        weight_stats.append({
            'layer': i + 1,
            'mean': float(np.mean(w)),
            'min': float(np.min(w)),
            'max': float(np.max(w)),
            'std': float(np.std(w)),
            'shape': list(w.shape)
        })
    
    bias_stats = []
    for i, b in enumerate(net.biases):
        bias_stats.append({
            'layer': i + 1,
            'mean': float(np.mean(b)),
            'min': float(np.min(b)),
            'max': float(np.max(b)),
            'std': float(np.std(b)),
            'shape': list(b.shape)
        })
    
    return jsonify({
        'network_id': network_id,
        'architecture': net.sizes,
        'trained': active_networks[network_id]['trained'],
        'accuracy': accuracy,
        'weight_stats': weight_stats,
        'bias_stats': bias_stats
    })

@app.route('/api/networks/<network_id>/successful_example', methods=['GET'])
def get_successful_example(network_id):
    """Return a random successful example prediction with network output details"""
    if network_id not in active_networks:
        return jsonify({'error': 'Network not found'}), 404
        
    net = active_networks[network_id]['network']
    
    # Find a successful example
    successful_example = None
    attempts = 0
    max_attempts = 100
    
    while successful_example is None and attempts < max_attempts:
        # Choose a random example from test data
        index = np.random.randint(0, len(test_data))
        x, y = test_data[index]
        
        # Get the network's output
        output = net.feedforward(x)
        predicted_digit = int(np.argmax(output))
        actual_digit = int(y)
        
        if predicted_digit == actual_digit:
            successful_example = {
                'index': index,
                'x': x,
                'y': actual_digit,
                'output': output,
                'predicted': predicted_digit
            }
        attempts += 1
    
    if successful_example is None:
        return jsonify({'error': 'No successful example found after multiple attempts'}), 404
    
    # Create image of the digit
    plt.figure(figsize=(3, 3))
    plt.imshow(successful_example['x'].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {successful_example['predicted']} | Actual: {successful_example['y']}")
    plt.axis('off')
    
    # Save image to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Get the output layer weights (last layer in the network)
    output_weights = net.weights[-1].tolist()
    
    return jsonify({
        'network_id': network_id,
        'example_index': successful_example['index'],
        'predicted_digit': successful_example['predicted'],
        'actual_digit': successful_example['y'],
        'image_data': img_base64,
        'output_weights': output_weights,
        'network_output': [float(val) for val in successful_example['output']]
    })

@app.route('/api/networks/<network_id>/unsuccessful_example', methods=['GET'])
def get_unsuccessful_example(network_id):
    """Return a random unsuccessful example prediction with network output details"""
    if network_id not in active_networks:
        return jsonify({'error': 'Network not found'}), 404
        
    net = active_networks[network_id]['network']
    
    # Find an unsuccessful example
    unsuccessful_example = None
    attempts = 0
    max_attempts = 200
    
    while unsuccessful_example is None and attempts < max_attempts:
        # Choose a random example from test data
        index = np.random.randint(0, len(test_data))
        x, y = test_data[index]
        
        # Get the network's output
        output = net.feedforward(x)
        predicted_digit = int(np.argmax(output))
        actual_digit = int(y)
        
        if predicted_digit != actual_digit:
            unsuccessful_example = {
                'index': index,
                'x': x,
                'y': actual_digit,
                'output': output,
                'predicted': predicted_digit
            }
        attempts += 1
    
    if unsuccessful_example is None:
        return jsonify({'error': 'No unsuccessful example found after multiple attempts'}), 404
    
    # Create image of the digit
    plt.figure(figsize=(3, 3))
    plt.imshow(unsuccessful_example['x'].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {unsuccessful_example['predicted']} | Actual: {unsuccessful_example['y']}")
    plt.axis('off')
    
    # Save image to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Get the output layer weights (last layer in the network)
    output_weights = net.weights[-1].tolist()
    
    return jsonify({
        'network_id': network_id,
        'example_index': unsuccessful_example['index'],
        'predicted_digit': unsuccessful_example['predicted'],
        'actual_digit': unsuccessful_example['y'],
        'image_data': img_base64,
        'output_weights': output_weights,
        'network_output': [float(val) for val in unsuccessful_example['output']]
    })

@app.route('/')
def index():
    """Serve the main frontend page"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    # Make sure the static directory exists
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    # Tell the user where the server is running
    print(f"Server running at http://localhost:8000/")
    
    # Set host to '0.0.0.0' to make it accessible from other machines
    app.run(host='0.0.0.0', debug=True, port=8000)
