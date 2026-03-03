"""
api_server.py
~~~~~~~~~~~~~

Flask-based REST API server with WebSocket support for neural network training.

This module provides endpoints for:
- Creating and managing neural networks
- Training networks with real-time progress updates via WebSockets
- Testing networks with MNIST digit recognition
- Persisting networks to/from SQLite database

The server uses:
- Flask for REST API endpoints
- Flask-SocketIO for WebSocket communication
- Gevent for async background training tasks
- SQLite for network persistence
"""

import os
import sys
import uuid
import json
import base64
import logging
from io import BytesIO
from typing import Dict, Any, List, Optional

import gevent
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

# OpenAI import (optional - only needed for about customization)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

# Use non-GUI backend for matplotlib (required for server environments)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
from src import network
from src import mnist_loader
from src.model_persistence import (
    save_network,
    load_network,
    list_saved_networks,
    delete_network,
    delete_old_networks
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def configure_logging() -> None:
    """
    Set up logging based on environment.

    - In production: Show fewer logs (less noise) but keep important logs
    - In development: Show more detailed logs for debugging
    """
    log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    is_production = os.getenv('FLASK_ENV') == 'production'

    # Set up basic logging format
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # In production, silence noisy third-party logs but keep our logs visible
    if is_production:
        for logger_name in ['socketio', 'engineio', 'engineio.server',
                            'socketio.server', 'werkzeug']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        # Keep our logs at INFO level for visibility in production
        logging.getLogger('src').setLevel(logging.INFO)
        logging.getLogger('src.api_server').setLevel(logging.INFO)
        logging.getLogger('src.model_persistence').setLevel(logging.INFO)
    else:
        logging.getLogger('socketio').setLevel(logging.INFO)
        logging.getLogger('engineio').setLevel(logging.INFO)


configure_logging()
logger = logging.getLogger(__name__)

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from any origin

is_production = os.getenv('FLASK_ENV') == 'production'

# SocketIO enables real-time communication (WebSockets) for training updates
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    logger=not is_production,
    engineio_logger=not is_production,
    ping_timeout=60,
    ping_interval=25
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Networks currently loaded in memory: {network_id: network_info}
active_networks: Dict[str, Dict[str, Any]] = {}

# Training jobs being tracked: {job_id: job_info}
training_jobs: Dict[str, Dict[str, Any]] = {}

# MNIST dataset - loaded once at startup for efficiency
# Each entry is a tuple of (image_data, label)
training_data: Any = None
validation_data: Any = None
test_data: Any = None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mnist_data() -> None:
    """
    Load MNIST dataset into global variables.

    Called once at startup to avoid reloading data for each training run.
    """
    global training_data, validation_data, test_data

    logger.info("Loading MNIST data...")
    try:
        training_data, validation_data, test_data = (
            mnist_loader.load_data_wrapper()
        )
        logger.info(
            f"Data loaded: {len(training_data)} training, "
            f"{len(validation_data)} validation, {len(test_data)} test"
        )
    except Exception as e:
        logger.exception(f"Error loading MNIST data: {e}")
        raise


def reload_saved_networks() -> None:
    """
    Reload all saved networks from the database into memory.

    Called at startup to restore networks that were saved before the
    application was restarted. This keeps active_networks in sync with
    the database.
    """
    saved_networks = list_saved_networks()

    if not saved_networks:
        logger.info("No saved networks to reload")
        return

    loaded_count = 0
    for net_info in saved_networks:
        network_id = net_info['network_id']
        try:
            net = load_network(network_id)
            if net is not None:
                active_networks[network_id] = {
                    'network': net,
                    'architecture': net_info['architecture'],
                    'trained': net_info['trained'],
                    'accuracy': net_info['accuracy']
                }
                loaded_count += 1
            else:
                logger.warning(f"Failed to load network {network_id}")
        except Exception as e:
            logger.exception(f"Error loading network {network_id}: {e}")

    logger.info(f"Reloaded {loaded_count} network(s) from database")


load_mnist_data()
reload_saved_networks()

# Clear any stale training jobs from before restart
# (Training jobs can't continue after a restart, so start fresh)
training_jobs.clear()
logger.info("Cleared stale training jobs")


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

# Flag to ensure cleanup task only starts once
_cleanup_task_started = False


def cleanup_old_networks_task() -> None:
    """
    Background task that runs immediately on startup, then every 24 hours to:
    - Delete networks older than 2 days from the database
    - Sync in-memory networks with the database
    - Remove completed/failed training jobs from memory

    Note: In production environments where the app may restart frequently,
    this ensures cleanup runs at least once per startup.
    """
    logger.info("=" * 60)
    logger.info("CLEANUP TASK STARTED")
    logger.info("=" * 60)

    while True:
        try:
            # Clean up old networks from database
            logger.info("Starting automatic cleanup of old networks...")
            logger.info(f"Active networks in memory before cleanup: {len(active_networks)}")

            deleted_count = delete_old_networks(days=2)

            if deleted_count > 0:
                logger.info(f"Cleanup completed: deleted {deleted_count} network(s)")

                # Sync in-memory networks with database
                # Remove any networks from memory that no longer exist in database
                saved_ids = {net['network_id'] for net in list_saved_networks()}
                networks_to_remove = [
                    nid for nid in active_networks.keys()
                    if nid not in saved_ids
                ]
                for nid in networks_to_remove:
                    del active_networks[nid]
                    logger.info(f"Removed network {nid} from memory (deleted from database)")

                logger.info(f"Active networks in memory after cleanup: {len(active_networks)}")
            elif deleted_count == 0:
                logger.info("Cleanup completed: no old networks found to delete")
            else:
                logger.error("Cleanup returned error code")

            # Clean up completed/failed training jobs from memory
            cleanup_finished_training_jobs()

            # Wait 24 hours before next cleanup
            logger.info("Next cleanup scheduled in 24 hours")
            gevent.sleep(86400)

        except Exception as e:
            logger.exception(f"Error during network cleanup: {e}")
            # Wait a bit before retrying on error (don't spam)
            gevent.sleep(3600)  # 1 hour
            continue


def cleanup_finished_training_jobs() -> None:
    """
    Remove completed or failed training jobs from memory.

    This prevents the training_jobs dictionary from growing indefinitely.
    Only removes jobs that are no longer active (completed or failed).
    """
    finished_statuses = {'completed', 'failed'}
    jobs_to_remove = [
        job_id for job_id, job_info in training_jobs.items()
        if job_info.get('status') in finished_statuses
    ]

    for job_id in jobs_to_remove:
        del training_jobs[job_id]

    if jobs_to_remove:
        logger.info(f"Cleaned up {len(jobs_to_remove)} finished training job(s)")


def start_cleanup_task() -> None:
    """
    Start the background cleanup task.

    Uses gevent.spawn() directly instead of socketio.start_background_task()
    to ensure it works both when running directly and under gunicorn.
    This function is idempotent - calling it multiple times has no effect.
    """
    global _cleanup_task_started

    if _cleanup_task_started:
        logger.debug("Cleanup task already started, skipping")
        return

    _cleanup_task_started = True
    logger.info("Starting cleanup task (runs immediately, then every 24 hours)")
    gevent.spawn(cleanup_old_networks_task)


# Start the cleanup task when module is loaded (works with gunicorn)
start_cleanup_task()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Return server status and statistics.

    Returns counts of active networks and training jobs that are
    currently in progress (status='pending' or 'training').
    """
    # Count only jobs that are actively training or pending
    active_statuses = ('pending', 'training')
    active_training = sum(
        1 for job in training_jobs.values()
        if job.get('status') in active_statuses
    )

    return jsonify({
        'status': 'online',
        'active_networks': len(active_networks),
        'training_jobs': active_training
    }), 200

@app.route('/api/networks', methods=['POST'])
def create_network():
    """
    Create a new neural network.

    Request body (optional):
        {'layer_sizes': [784, 30, 10]}  # defaults to [784, 30, 10]

    Returns:
        JSON with network_id, architecture, and status
    """
    data = request.get_json() or {}
    layer_sizes = data.get('layer_sizes', [784, 30, 10])

    # Validate: need at least input and output layers
    if not isinstance(layer_sizes, list) or len(layer_sizes) < 2:
        logger.warning(f"Invalid architecture requested: {layer_sizes}")
        return jsonify({
            'error': 'Invalid architecture. Must have at least 2 layers.'
        }), 400

    network_id = str(uuid.uuid4())

    try:
        net = network.Network(layer_sizes)

        active_networks[network_id] = {
            'network': net,
            'architecture': layer_sizes,
            'trained': False,
            'accuracy': None
        }

        logger.info(f"Created network {network_id} with architecture {layer_sizes}")

        return jsonify({
            'network_id': network_id,
            'architecture': layer_sizes,
            'status': 'created'
        }), 201

    except Exception as e:
        logger.exception(f"Error creating network: {e}")
        return jsonify({'error': f'Failed to create network: {str(e)}'}), 500

@app.route('/api/networks/<network_id>/train', methods=['POST'])
def train_network(network_id: str):
    """
    Start training a network in the background.

    Request body (all optional):
        {
            'epochs': 5,
            'mini_batch_size': 10,
            'learning_rate': 3.0
        }

    Returns:
        JSON with job_id, network_id, and status
    """
    if network_id not in active_networks:
        logger.warning(f"Training requested for non-existent network: {network_id}")
        return jsonify({'error': 'Network not found'}), 404

    data = request.get_json() or {}
    epochs = data.get('epochs', 5)
    mini_batch_size = data.get('mini_batch_size', 10)
    learning_rate = data.get('learning_rate', 3.0)

    # Validate training parameters
    if not isinstance(epochs, int) or epochs < 1:
        return jsonify({'error': 'epochs must be a positive integer'}), 400
    if not isinstance(mini_batch_size, int) or mini_batch_size < 1:
        return jsonify({'error': 'mini_batch_size must be a positive integer'}), 400
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        return jsonify({'error': 'learning_rate must be a positive number'}), 400

    job_id = str(uuid.uuid4())

    training_jobs[job_id] = {
        'network_id': network_id,
        'status': 'pending',
        'progress': 0,
        'epochs': epochs
    }

    logger.info(
        f"Created training job {job_id} for network {network_id}: "
        f"epochs={epochs}, batch_size={mini_batch_size}, lr={learning_rate}"
    )

    # Run training in background so we can return immediately
    socketio.start_background_task(
        train_network_task,
        network_id, job_id, epochs, mini_batch_size, learning_rate
    )

    return jsonify({
        'job_id': job_id,
        'network_id': network_id,
        'status': 'training_started'
    }), 202

def train_network_task(
    network_id: str,
    job_id: str,
    epochs: int,
    mini_batch_size: int,
    learning_rate: float
) -> None:
    """
    Background task that trains a neural network.

    Sends progress updates via WebSocket as training progresses.
    """
    net = active_networks[network_id]['network']

    def on_epoch_complete(data: Dict[str, Any]) -> None:
        """Called after each training epoch to send progress updates."""
        progress = (data['epoch'] / data['total_epochs']) * 100

        training_jobs[job_id]['status'] = 'training'
        training_jobs[job_id]['progress'] = progress


        # Send update to connected clients via WebSocket
        socketio.emit('training_update', {
            'job_id': job_id,
            'network_id': network_id,
            'epoch': data['epoch'],
            'total_epochs': data['total_epochs'],
            'accuracy': data['accuracy'],
            'elapsed_time': data['elapsed_time'],
            'progress': progress,
            'correct': data.get('correct'),
            'total': data.get('total')
        })

        # Let gevent send the message immediately
        gevent.sleep(0)

    try:
        logger.info(f"Starting training for job {job_id}")

        # Define yield function for cooperative multitasking
        # This allows HTTP requests to be processed during training
        def yield_to_other_tasks():
            gevent.sleep(0)

        # Train the network
        net.SGD(
            training_data,
            epochs,
            mini_batch_size,
            learning_rate,
            test_data=test_data,
            callback=on_epoch_complete,
            yield_func=yield_to_other_tasks
        )

        # Calculate final accuracy
        accuracy = net.evaluate(test_data) / len(test_data)

        # Update network and job status
        active_networks[network_id]['trained'] = True
        active_networks[network_id]['accuracy'] = accuracy

        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['accuracy'] = accuracy
        training_jobs[job_id]['progress'] = 100

        # Save the trained network
        save_network(net, network_id, trained=True, accuracy=accuracy)

        logger.info(f"Training completed for job {job_id}: accuracy {accuracy:.2%}")

        # Notify clients that training is complete
        socketio.emit('training_complete', {
            'job_id': job_id,
            'network_id': network_id,
            'status': 'completed',
            'accuracy': float(accuracy),
            'progress': 100
        })
        gevent.sleep(0)

        # Clean up the job immediately (clients use WebSocket events, not polling)
        if job_id in training_jobs:
            del training_jobs[job_id]
            logger.debug(f"Cleaned up completed training job {job_id}")

    except Exception as e:
        logger.exception(f"Training failed for job {job_id}: {e}")

        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)

        socketio.emit('training_error', {
            'job_id': job_id,
            'network_id': network_id,
            'status': 'failed',
            'error': str(e)
        })
        gevent.sleep(0)

        # Clean up failed job immediately
        if job_id in training_jobs:
            del training_jobs[job_id]
            logger.debug(f"Cleaned up failed training job {job_id}")

@app.route('/api/training/<job_id>', methods=['GET'])
def get_training_status(job_id: str):
    """
    Get the current status of a training job.

    If the job doesn't exist but a trained network exists, returns completed status.
    """
    # If job exists in memory, return its status
    if job_id in training_jobs:
        return jsonify(training_jobs[job_id]), 200

    # Job not in memory - check if any network is trained (job may have been cleaned up)
    # Try to find a trained network (use job_id as potential network_id)
    if job_id in active_networks and active_networks[job_id].get('trained'):
        # Return a synthetic completed status
        return jsonify({
            'network_id': job_id,
            'status': 'completed',
            'progress': 100,
            'accuracy': active_networks[job_id].get('accuracy')
        }), 200

    # Check if there's any trained network at all and return generic completed status
    for network_id, net_info in active_networks.items():
        if net_info.get('trained'):
            return jsonify({
                'network_id': network_id,
                'status': 'completed',
                'progress': 100,
                'accuracy': net_info.get('accuracy'),
                'message': 'Training job completed and cleaned up'
            }), 200

    logger.warning(f"Status requested for non-existent job: {job_id}")
    return jsonify({'error': 'Training job not found'}), 404


@app.route('/api/networks', methods=['GET'])
def list_networks():
    """List all available networks (both in-memory and saved to disk)."""
    # Get networks currently in memory
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

    # Get saved networks, excluding duplicates already in memory
    in_memory_ids = set(active_networks.keys())
    saved = list_saved_networks()
    saved_only = []
    for net in saved:
        if net['network_id'] not in in_memory_ids:
            net['status'] = 'saved'
            saved_only.append(net)

    logger.debug(f"Listing networks: {len(in_memory)} in memory, {len(saved_only)} saved")

    return jsonify({'networks': in_memory + saved_only}), 200

@app.route('/api/networks/<network_id>', methods=['DELETE'])
def delete_network_endpoint(network_id: str):
    """Delete a network from both memory and disk."""
    deleted_from_memory = False
    if network_id in active_networks:
        del active_networks[network_id]
        deleted_from_memory = True

    deleted_from_disk = delete_network(network_id)

    if not deleted_from_memory and not deleted_from_disk:
        logger.warning(f"Delete attempted for non-existent network: {network_id}")
        return jsonify({'error': 'Network not found'}), 404

    logger.info(f"Deleted network {network_id}: memory={deleted_from_memory}, disk={deleted_from_disk}")

    return jsonify({
        'network_id': network_id,
        'deleted_from_memory': deleted_from_memory,
        'deleted_from_disk': deleted_from_disk
    }), 200

@app.route('/api/networks', methods=['DELETE'])
def delete_all_networks():
    """Delete all networks from both memory and disk."""
    in_memory_ids = list(active_networks.keys())
    saved_ids = [net['network_id'] for net in list_saved_networks()]
    all_network_ids = list(set(in_memory_ids + saved_ids))

    deleted_from_memory_count = 0
    deleted_from_disk_count = 0

    for network_id in all_network_ids:
        if network_id in active_networks:
            del active_networks[network_id]
            deleted_from_memory_count += 1

        if delete_network(network_id):
            deleted_from_disk_count += 1

    logger.info(
        f"Deleted all networks: {len(all_network_ids)} total, "
        f"{deleted_from_memory_count} from memory, {deleted_from_disk_count} from disk"
    )

    return jsonify({
        'deleted_count': len(all_network_ids),
        'deleted_from_memory': deleted_from_memory_count,
        'deleted_from_disk': deleted_from_disk_count,
        'message': f'Successfully deleted {len(all_network_ids)} network(s)'
    }), 200


@app.route('/api/networks/cleanup', methods=['POST'])
def cleanup_old_networks_endpoint():
    """
    Manually trigger cleanup of networks older than specified days.

    Request body (optional):
        {'days': 2}  # defaults to 2

    Returns:
        JSON with deleted_count, days, and message
    """
    data = request.get_json() or {}
    days = data.get('days', 2)

    if not isinstance(days, (int, float)) or days < 0:
        return jsonify({'error': 'days must be a non-negative number'}), 400

    try:
        deleted_count = delete_old_networks(days=int(days))

        if deleted_count == -1:
            return jsonify({'error': 'Error occurred during cleanup'}), 500

        logger.info(f"Manual cleanup: deleted {deleted_count} network(s) older than {days} day(s)")

        return jsonify({
            'deleted_count': deleted_count,
            'days': days,
            'message': f'Successfully deleted {deleted_count} network(s) older than {days} day(s)'
        }), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Error during manual cleanup: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# ABOUT ME CUSTOMIZATION
# ============================================================================

def load_work_history() -> Optional[str]:
    """
    Load the work history text file used as context for AI customization.

    Reads from data/work_history.txt at request time so edits to the
    file take effect without restarting the server.

    Returns:
        The raw text content of the work history file, or None if
        the file is missing or empty.
    """
    work_history_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'data', 'work_history.txt'
    )
    try:
        with open(work_history_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content if content else None
    except FileNotFoundError:
        logger.warning(f"Work history file not found: {work_history_path}")
        return None


@app.route('/api/about/customize', methods=['POST'])
def customize_about():
    """
    Generate a customized About Me and Skills section using OpenAI.

    Uses the contents of data/work_history.txt as grounding context
    combined with the user's prompt to generate personalized content.
    Only facts present in the work history file are used to prevent
    hallucinations.

    Request body:
        {'prompt': 'Make it sound more technical'}

    Returns:
        JSON with about_me (string with paragraphs separated by
        newlines) and skills (list of {category, tags} objects).
    """
    data = request.get_json() or {}
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({'error': 'prompt is required'}), 400

    # Load work history for grounding
    work_history = load_work_history()
    if not work_history:
        logger.warning("About customization requested but work history is empty")
        return jsonify({
            'error': 'Work history file is not configured. '
                     'Please populate data/work_history.txt.'
        }), 503

    # Check that OpenAI package is available
    if OpenAI is None:
        logger.error("openai package is not installed")
        return jsonify({
            'error': 'OpenAI package is not installed on the server.'
        }), 500

    # Check for OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        return jsonify({
            'error': 'OpenAI API key is not configured on the server.'
        }), 500

    system_prompt = (
        "You are a professional bio writer. You will be given a detailed "
        "work history document and a user request. Your job is to rewrite "
        "an 'About Me' biography and a 'Skills & Technologies' section "
        "based STRICTLY on the facts in the work history document.\n\n"
        "CRITICAL RULES:\n"
        "- Use ONLY facts, roles, companies, skills, and experiences "
        "that appear in the work history document.\n"
        "- Do NOT invent, fabricate, or assume any information not "
        "explicitly stated in the document.\n"
        "- If the user asks for something not supported by the work "
        "history, politely note that in the about_me text.\n"
        "- Write in third person or first person based on the user's "
        "preference. Default to first person.\n\n"
        "Return a JSON object with exactly two keys:\n"
        "1. \"about_me\": A string containing 2-4 paragraphs separated "
        "by \\n\\n (double newline). Plain text only, no markdown or HTML.\n"
        "2. \"skills\": An array of objects, each with:\n"
        "   - \"category\": A short category name (e.g., \"Frontend\", "
        "\"Backend\", \"Leadership\")\n"
        "   - \"tags\": An array of skill/technology strings belonging "
        "to that category\n\n"
        "WORK HISTORY DOCUMENT:\n"
        "---\n"
        f"{work_history}\n"
        "---"
    )

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=2048,
            temperature=0.7
        )

        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        # Validate response structure
        if 'about_me' not in result or 'skills' not in result:
            logger.error(f"Invalid AI response structure: {result.keys()}")
            return jsonify({
                'error': 'AI returned an unexpected response format.'
            }), 500

        if not isinstance(result['about_me'], str):
            return jsonify({
                'error': 'AI returned invalid about_me format.'
            }), 500

        if not isinstance(result['skills'], list):
            return jsonify({
                'error': 'AI returned invalid skills format.'
            }), 500

        logger.info("About Me customization generated successfully")

        return jsonify({
            'about_me': result['about_me'],
            'skills': result['skills']
        }), 200

    except json.JSONDecodeError as e:
        logger.exception(f"Failed to parse AI response as JSON: {e}")
        return jsonify({
            'error': 'AI returned an invalid response. Please try again.'
        }), 500
    except Exception as e:
        logger.exception(f"Error during about customization: {e}")
        return jsonify({
            'error': f'Failed to generate customized content: {str(e)}'
        }), 500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def array_to_float_list(array: np.ndarray) -> List[float]:
    """Convert a numpy array to a list of floats (for JSON serialization)."""
    return [float(val) for val in array.flatten()]


def create_digit_image(image_data: np.ndarray, predicted: int, actual: int) -> str:
    """
    Create a base64-encoded PNG image of a digit.

    Args:
        image_data: 784-element array representing the 28x28 digit image
        predicted: The digit the network predicted (0-9)
        actual: The correct digit (0-9)

    Returns:
        Base64-encoded PNG image string
    """
    plt.figure(figsize=(3, 3))
    plt.imshow(image_data.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted} | Actual: {actual}")
    plt.axis('off')

    # Save image to a bytes buffer instead of a file
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return img_base64


# ============================================================================
# EXAMPLE ENDPOINTS
# ============================================================================

@app.route('/api/networks/<network_id>/successful_example', methods=['GET'])
def get_successful_example(network_id: str):
    """
    Find and return a random example where the network predicted correctly.

    Returns JSON with image, prediction details, and network output.
    """
    if network_id not in active_networks:
        logger.warning(f"Successful example requested for non-existent network: {network_id}")
        return jsonify({'error': 'Network not found'}), 404

    if test_data is None:
        logger.error("Test data not loaded")
        return jsonify({'error': 'Test data not available'}), 500

    net = active_networks[network_id]['network']
    data = test_data  # Local reference after None check

    # Try to find a successful prediction (max 100 attempts)
    max_attempts = 100
    for attempt in range(max_attempts):
        index = np.random.randint(0, len(data))
        x, y = data[index]

        output = net.feedforward(x)
        predicted_digit = int(np.argmax(output))
        actual_digit = int(y)

        if predicted_digit == actual_digit:
            logger.debug(f"Found successful example on attempt {attempt + 1}")

            return jsonify({
                'network_id': network_id,
                'example_index': index,
                'predicted_digit': predicted_digit,
                'actual_digit': actual_digit,
                'image_data': create_digit_image(x, predicted_digit, actual_digit),
                'output_weights': net.weights[-1].tolist(),
                'network_output': array_to_float_list(output)
            }), 200

    logger.warning(f"No successful example found after {max_attempts} attempts")
    return jsonify({
        'error': f'No successful example found after {max_attempts} attempts'
    }), 404

@app.route('/api/networks/<network_id>/unsuccessful_example', methods=['GET'])
def get_unsuccessful_example(network_id: str):
    """
    Find and return a random example where the network predicted incorrectly.

    Returns JSON with image, prediction details, and network output.
    """
    if network_id not in active_networks:
        logger.warning(f"Unsuccessful example requested for non-existent network: {network_id}")
        return jsonify({'error': 'Network not found'}), 404

    if test_data is None:
        logger.error("Test data not loaded")
        return jsonify({'error': 'Test data not available'}), 500

    net = active_networks[network_id]['network']
    data = test_data  # Local reference after None check

    # Try to find an unsuccessful prediction (max 200 attempts)
    max_attempts = 200
    for attempt in range(max_attempts):
        index = np.random.randint(0, len(data))
        x, y = data[index]

        output = net.feedforward(x)
        predicted_digit = int(np.argmax(output))
        actual_digit = int(y)

        if predicted_digit != actual_digit:
            logger.debug(f"Found unsuccessful example on attempt {attempt + 1}")

            return jsonify({
                'network_id': network_id,
                'example_index': index,
                'predicted_digit': predicted_digit,
                'actual_digit': actual_digit,
                'image_data': create_digit_image(x, predicted_digit, actual_digit),
                'output_weights': net.weights[-1].tolist(),
                'network_output': array_to_float_list(output)
            }), 200

    logger.warning(f"No unsuccessful example found after {max_attempts} attempts")
    return jsonify({
        'error': f'No unsuccessful example found after {max_attempts} attempts'
    }), 404

# ============================================================================
# STATIC FILE SERVING
# ============================================================================

@app.route('/')
def index():
    """Serve the main frontend page."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path: str):
    """Serve static files (CSS, JS, images, etc.)."""
    return send_from_directory(app.static_folder, path)

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        logger.info(f"Created static directory: {static_dir}")

    # Check if running in cloud environment (Railway, etc.)
    is_cloud = bool(os.environ.get('RAILWAY_STATIC_URL') or os.environ.get('PORT'))
    port = int(os.environ.get('PORT', 8000))

    if is_cloud:
        logger.info(f"Starting server in production mode on port {port}")
    else:
        logger.info(f"Starting server at http://localhost:{port}/")

    # Start background cleanup task
    start_cleanup_task()

    # Start the server with WebSocket support
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=not is_cloud,
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is already in use.")
            logger.info("You can use: pkill -f 'python src/api_server.py'")
            sys.exit(1)
        else:
            raise
