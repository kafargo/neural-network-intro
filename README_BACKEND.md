# Neural Network Backend System

This project is a backend system for creating, training, and using neural networks. The system allows a frontend application to:

1. Create a neural network with a customizable architecture
2. Train the network with a specified number of epochs and parameters
3. Test the network on MNIST data
4. Visualize the network structure and results

## Architecture

The system consists of:

- **Flask API Server**: Handles requests from frontend applications
- **Neural Network Module**: Implements the neural network functionality
- **Model Persistence**: Allows saving and loading trained networks
- **Web Interface**: Simple frontend for interacting with the API

## Getting Started

### Accessing the Server

The server is configured to run on all network interfaces (`0.0.0.0`), which means you can access it:

- From the same machine using: `http://localhost:8000/`
- From other devices on the same network using: `http://<your-machine-ip>:8000/`

You can run `./access_info.sh` to see all available ways to access your server.

> **Note**: If you're having trouble accessing the server from another device, check your firewall settings to ensure port 8000 is allowed.

> **Note for macOS users**: We're using port 8000 instead of 5000 because port 5000 is commonly used by AirPlay Receiver on macOS.

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:

```
git clone https://github.com/yourusername/neural-network-backend.git
cd neural-network-backend
```

2. Run the server script:

```
./run_server.sh
```

This will:

- Create a virtual environment (if it doesn't exist)
- Install required packages
- Start the API server on port 5000

### Usage

Once the server is running, you can:

1. Access the web interface at http://localhost:5000/
2. Use the API directly for custom frontend development

## API Endpoints

The following API endpoints are available:

### Network Management

- `GET /api/status`: Get server status
- `POST /api/networks`: Create a new neural network
- `GET /api/networks`: List all available networks
- `DELETE /api/networks/{network_id}`: Delete a network
- `POST /api/networks/{network_id}/load`: Load a saved network into memory

### Training

- `POST /api/networks/{network_id}/train`: Start training a network
- `GET /api/training/{job_id}`: Check training status

### Testing and Visualization

- `POST /api/networks/{network_id}/predict`: Run prediction on a single example
- `POST /api/networks/{network_id}/predict_batch`: Run predictions on multiple examples
- `GET /api/networks/{network_id}/misclassified`: Find misclassified examples
- `GET /api/networks/{network_id}/visualize`: Get network visualization
- `GET /api/networks/{network_id}/stats`: Get network statistics
- `GET /api/networks/{network_id}/successful_example`: Get a random successful prediction with visualization and output weights
- `GET /api/networks/{network_id}/unsuccessful_example`: Get a random unsuccessful prediction with visualization and output weights

## Troubleshooting

### Connection Issues

If you're experiencing connection issues to the server:

1. **Port 5000 conflicts (macOS)**: We now use port 8000 by default because port 5000 is used by AirPlay Receiver on macOS.

   - If you still see port conflicts, try: `System Settings → AirDrop & Handoff → Turn off AirPlay Receiver`.

2. **"Access to localhost was denied" error**:

   - Ensure you're accessing the server using the correct hostname/IP and port (8000)
   - If accessing from another device, use `http://<your-machine-ip>:8000` instead of `localhost`
   - Check that the server is running and listening on all interfaces (`0.0.0.0`)
   - Verify that your firewall is not blocking connections

3. **Cross-Origin Issues**:

   - The server has CORS (Cross-Origin Resource Sharing) enabled
   - If you're still seeing CORS errors, ensure your browser accepts connections from your server

4. **Viewing Available Connection Options**:
   Run: `./access_info.sh` to see all available IP addresses and connection URLs

### Server Administration

1. **Starting the server with fixed imports**:

   - Use `./fix_and_run.sh` to correctly configure imports and run the server

2. **Checking if the server is running**:
   - `curl http://localhost:8000/api/status` should return a JSON response
3. **Testing the API**:
   - Use `./test_api.sh` to test all main API endpoints
   - Use `./test_new_endpoints.sh` to test the example endpoints

## Frontend Integration

To integrate with your own frontend application:

1. Make HTTP requests to the API endpoints above
2. Process the JSON responses according to your application's needs

Example code for creating a network:

```javascript
async function createNetwork(layerSizes) {
  const response = await fetch("http://localhost:5000/api/networks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ layer_sizes: layerSizes }),
  });
  return await response.json();
}
```

## Original Project

This backend system is built on top of the neural networks code from Michael Nielsen's book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com).

## License

This project is licensed under the MIT License.
