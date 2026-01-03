# Neural Network Backend System

A clean, focused backend API for creating and training neural networks on MNIST digit recognition. Built with Flask and Flask-SocketIO to provide real-time training updates to frontend applications.

## Features

This system allows frontend applications to:

1. **Create** neural networks with customizable architectures
2. **Train** networks with configurable parameters (epochs, batch size, learning rate)
3. **Monitor** training progress in real-time via WebSocket updates
4. **Test** networks with successful and unsuccessful example predictions
5. **Manage** networks (list, delete, automatic persistence)

## Architecture

The system consists of:

- **Flask API Server** (`api_server.py`): RESTful HTTP endpoints + WebSocket support
- **Neural Network Module** (`network.py`): Core backpropagation implementation
- **MNIST Loader** (`mnist_loader.py`): Handles MNIST dataset loading
- **Model Persistence** (`model_persistence.py`): Automatic save/load of trained networks
- **Simple Landing Page** (`static/index.html`): Basic network management interface

## Getting Started

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/neural-networks-and-deep-learning.git
cd neural-networks-and-deep-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
python src/api_server.py
```

The server will start on **port 8000** and be accessible at:
- Local: `http://localhost:8000/`
- Network: `http://<your-machine-ip>:8000/`

> **Note**: Port 8000 is used instead of 5000 because port 5000 is commonly used by AirPlay Receiver on macOS.

### Quick Test

Once running, test the API:
```bash
# Check server status
curl http://localhost:8000/api/status

# Create a test network
curl -X POST http://localhost:8000/api/networks \
  -H "Content-Type: application/json" \
  -d '{"layer_sizes": [784, 30, 10]}'
```

### Access the Management Interface

Visit `http://localhost:8000/` in your browser to see the simple network management page where you can:
- View all created networks
- See training status and accuracy
- Delete networks

## API Endpoints

### Core Endpoints

#### Status & Health
- **GET /api/status**  
  Check if the API is running and get basic statistics
  
#### Network Management
- **POST /api/networks**  
  Create a new neural network with specified architecture  
  Body: `{"layer_sizes": [784, 30, 10]}`
  
- **GET /api/networks**  
  List all available networks (both in-memory and saved)
  
- **DELETE /api/networks/{network_id}**  
  Delete a specific network from both memory and disk

- **DELETE /api/networks**  
  Delete all networks (both in-memory and saved)

#### Training
- **POST /api/networks/{network_id}/train**  
  Start asynchronous training for a network  
  Body: `{"epochs": 5, "mini_batch_size": 10, "learning_rate": 3.0}` (defaults shown)
  
- **GET /api/training/{job_id}**  
  Get the status of a training job

#### Testing & Examples
- **GET /api/networks/{network_id}/successful_example**  
  Get a random example that the network predicted correctly  
  Returns: digit image (base64), prediction, actual label, network outputs
  
- **GET /api/networks/{network_id}/unsuccessful_example**  
  Get a random example that the network predicted incorrectly  
  Returns: digit image (base64), prediction, actual label, network outputs

### WebSocket Events

#### Real-Time Training Updates
Connect to the WebSocket server to receive live training progress:

```javascript
const socket = io('http://localhost:8000');

socket.on('training_update', (data) => {
  console.log(`Epoch ${data.epoch}/${data.total_epochs}`);
  console.log(`Accuracy: ${(data.accuracy * 100).toFixed(2)}%`);
  console.log(`Progress: ${data.progress}%`);
});

socket.on('training_complete', (data) => {
  console.log('Training completed!');
  console.log(`Final accuracy: ${(data.accuracy * 100).toFixed(2)}%`);
});

socket.on('training_error', (data) => {
  console.error('Training failed:', data.error);
});
```

**Available Events:**
- `training_update` - Emitted after each epoch with progress data
- `training_complete` - Emitted when training finishes successfully
- `training_error` - Emitted if training fails with error details

**Event payload for `training_update` includes:**
- `job_id`: Training job identifier
- `network_id`: Network being trained
- `epoch`: Current epoch number
- `total_epochs`: Total epochs to train
- `accuracy`: Current test accuracy (0-1)
- `progress`: Percentage complete (0-100)
- `elapsed_time`: Seconds elapsed
- `correct`: Number of correct predictions
- `total`: Total test examples

## Troubleshooting

### Connection Issues

**Port Conflicts (macOS)**  
We use port 8000 by default. If you see port conflicts:
- Port 5000 is used by AirPlay Receiver on macOS
- You can disable it: `System Settings → AirDrop & Handoff → Turn off AirPlay Receiver`

**"Access denied" or Connection Refused**
- Ensure the server is running: `curl http://localhost:8000/api/status`
- If accessing from another device, use `http://<your-machine-ip>:8000`
- Check that your firewall allows connections on port 8000

**CORS Issues**
- The server has CORS enabled for all origins (`*`)
- If issues persist, check browser console for specific errors

### Server Management

**Check if server is running:**
```bash
curl http://localhost:8000/api/status
```

**List all networks:**
```bash
curl http://localhost:8000/api/networks
```

**Kill stuck server process:**
```bash
pkill -f 'python src/api_server.py'
```

## Frontend Integration

### HTTP API Integration

Example code for creating and training a network:

```javascript
// Create a network
async function createNetwork(layerSizes) {
  const response = await fetch("http://localhost:8000/api/networks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ layer_sizes: layerSizes }),
  });
  return await response.json();
}

// Start training
async function trainNetwork(networkId, epochs = 5) {
  const response = await fetch(
    `http://localhost:8000/api/networks/${networkId}/train`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: epochs,
        mini_batch_size: 10,
        learning_rate: 3.0,
      }),
    }
  );
  return await response.json();
}
```

### WebSocket Integration

Connect to receive real-time training updates:

```javascript
import { io } from "socket.io-client";

const socket = io("http://localhost:8000");

// Listen for training updates
socket.on("training_update", (data) => {
  console.log(`Epoch ${data.epoch}/${data.total_epochs}`);
  console.log(`Accuracy: ${(data.accuracy * 100).toFixed(2)}%`);
  updateProgressBar(data.progress);
});

// Handle training completion
socket.on("training_complete", (data) => {
  console.log("Training completed successfully!");
  console.log(`Final accuracy: ${(data.accuracy * 100).toFixed(2)}%`);
});

// Handle training errors
socket.on("training_error", (data) => {
  console.error("Training failed:", data.error);
});

// Handle connection events
socket.on("connect", () => console.log("Connected to server"));
socket.on("disconnect", () => console.log("Disconnected from server"));
```

### Example: Full Training Flow

```javascript
async function trainWithUpdates() {
  // 1. Create network
  const network = await createNetwork([784, 128, 10]);
  console.log("Network created:", network.network_id);

  // 2. Connect WebSocket
  const socket = io("http://localhost:8000");

  // 3. Listen for updates
  socket.on("training_update", (data) => {
    if (data.network_id === network.network_id) {
      console.log(`Progress: ${data.progress.toFixed(1)}%`);
      console.log(`Accuracy: ${(data.accuracy * 100).toFixed(2)}%`);
    }
  });

  socket.on("training_complete", (data) => {
    if (data.network_id === network.network_id) {
      console.log("Training completed!");
      console.log(`Final accuracy: ${(data.accuracy * 100).toFixed(2)}%`);
    }
  });

  socket.on("training_error", (data) => {
    if (data.network_id === network.network_id) {
      console.error("Training failed:", data.error);
    }
  });

  // 4. Start training
  const job = await trainNetwork(network.network_id, 10);
  console.log("Training started:", job.job_id);
}
```

## Project Structure

```
neural-networks-and-deep-learning/
├── src/
│   ├── api_server.py          # Main Flask + SocketIO server
│   ├── network.py              # Neural network implementation
│   ├── mnist_loader.py         # MNIST data loader
│   ├── model_persistence.py    # Save/load networks
│   └── static/
│       └── index.html          # Simple management UI
├── docs/
│   ├── API_DOCUMENTATION.md           # Complete REST API docs
│   ├── WEBSOCKET_API_DOCUMENTATION.md # WebSocket events & Angular guide
│   └── README_ORIGINAL.md             # Original project documentation
├── models/                     # Saved trained networks (created on first save)
├── data/
│   └── mnist.pkl.gz           # MNIST dataset
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Docker Compose setup
├── railway.json                # Railway deployment configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Documentation

For detailed API information, see the `docs/` folder:

- **[docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)** - Complete REST API documentation with examples
- **[docs/WEBSOCKET_API_DOCUMENTATION.md](docs/WEBSOCKET_API_DOCUMENTATION.md)** - WebSocket events and Angular integration guide
- **[docs/README_ORIGINAL.md](docs/README_ORIGINAL.md)** - Original project documentation

## Original Project

This backend system is built on top of the neural networks code from Michael Nielsen's book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com).

## License

This project is licensed under the MIT License.
