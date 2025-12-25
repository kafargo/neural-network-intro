"""
api_server.py
~~~~~~~~~~~~~~~~~

A Flask-based API server that serves as the backend for the neural network
application. It enables frontend applications to create networks, train them,
and retrieve results.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
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
    from model_persistence import (
        save_network,
        load_network,
        list_saved_networks,
        delete_network
    )

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and all origins
socketio = SocketIO(app, cors_allowed_origins="*")

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
    
    def epoch_callback(data):
        """Callback function for each epoch to send updates via websocket"""
        # Update the job status
        training_jobs[job_id]['status'] = 'training'
        training_jobs[job_id]['progress'] = (data['epoch'] / data['total_epochs']) * 100
        
        # Prepare update data for websocket emission
        update_data = {
            'job_id': job_id,
            'network_id': network_id,
            'epoch': data['epoch'],
            'total_epochs': data['total_epochs'],
            'accuracy': data['accuracy'],
            'elapsed_time': data['elapsed_time'],
            'progress': training_jobs[job_id]['progress'],
            'correct': data.get('correct'),
            'total': data.get('total')
        }
        
        # Emit the progress update through websocket
        socketio.emit('training_update', update_data)
    
    try:
        # Train the network with the callback function
        net.SGD(training_data, epochs, mini_batch_size, learning_rate, 
                test_data=test_data, callback=epoch_callback)
        
        # Calculate accuracy
        accuracy = net.evaluate(test_data) / len(test_data)
        
        # Update network status
        active_networks[network_id]['trained'] = True
        active_networks[network_id]['accuracy'] = accuracy
        
        # Update job status
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['accuracy'] = accuracy
        training_jobs[job_id]['progress'] = 100

        # Save the trained network with metadata
        save_network(net, network_id, trained=True, accuracy=accuracy)

        # Emit completion event via WebSocket
        socketio.emit('training_complete', {
            'job_id': job_id,
            'network_id': network_id,
            'status': 'completed',
            'accuracy': float(accuracy),
            'progress': 100
        })

    except Exception as e:
        # Update job status on error
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)

        # Emit error event via WebSocket
        socketio.emit('training_error', {
            'job_id': job_id,
            'network_id': network_id,
            'status': 'failed',
            'error': str(e)
        })

@app.route('/api/training/<job_id>', methods=['GET'])
def get_training_status(job_id):
    """Get the status of a training job"""
    if job_id not in training_jobs:
        return jsonify({'error': 'Training job not found'}), 404
    
    return jsonify(training_jobs[job_id])

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
    
    # Get saved networks, but exclude ones already in memory to avoid duplicates
    in_memory_ids = set(active_networks.keys())
    saved = list_saved_networks()
    saved_not_in_memory = []
    for net in saved:
        if net['network_id'] not in in_memory_ids:
            net['status'] = 'saved'
            saved_not_in_memory.append(net)

    return jsonify({
        'networks': in_memory + saved_not_in_memory
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

@app.route('/api/networks', methods=['DELETE'])
def delete_all_networks():
    """Delete all networks (both from memory and disk)"""
    # Get all network IDs (in-memory and saved)
    in_memory_ids = list(active_networks.keys())
    saved_networks = list_saved_networks()
    saved_ids = [net['network_id'] for net in saved_networks]

    # Combine and deduplicate
    all_network_ids = list(set(in_memory_ids + saved_ids))

    deleted_count = 0
    deleted_from_memory_count = 0
    deleted_from_disk_count = 0

    # Delete each network
    for network_id in all_network_ids:
        # Remove from active networks if present
        if network_id in active_networks:
            del active_networks[network_id]
            deleted_from_memory_count += 1

        # Delete from disk if present
        if delete_network(network_id):
            deleted_from_disk_count += 1

        deleted_count += 1

    return jsonify({
        'deleted_count': deleted_count,
        'deleted_from_memory': deleted_from_memory_count,
        'deleted_from_disk': deleted_from_disk_count,
        'message': f'Successfully deleted {deleted_count} network(s)'
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
    
    # Check if we're running in a cloud environment like Railway
    is_production = os.environ.get('RAILWAY_STATIC_URL', False) or os.environ.get('PORT', False)
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8000))
    
    # Tell the user where the server is running
    if is_production:
        print(f"Server running in production mode on port {port}")
    else:
        print(f"Server running at http://localhost:{port}/")
    
    # Use SocketIO for running the app instead of regular Flask
    # This enables WebSocket support
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=not is_production,
            use_reloader=False,
            allow_unsafe_werkzeug=True  # Required for newer versions of Flask-SocketIO
        )
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Error: Port {port} is already in use.")
            print("Please terminate the other server process before starting a new one.")
            print("You can use the command: pkill -f 'python src/api_server.py'")
            import sys
            sys.exit(1)
        else:
            raise
