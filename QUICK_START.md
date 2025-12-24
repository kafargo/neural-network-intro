# Quick Start Guide

After the cleanup, here's how to use your streamlined neural network API.

## Starting the Server

```bash
# From the project root directory
python src/api_server.py
```

The server will start on `http://localhost:8000`

## Testing the API

### Option 1: Use the Test Script
```bash
# In another terminal, run:
python test_api.py
```

This will test all essential endpoints automatically.

### Option 2: Manual Testing with cURL

**Check API Status:**
```bash
curl http://localhost:8000/api/status
```

**Create a Network:**
```bash
curl -X POST http://localhost:8000/api/networks \
  -H "Content-Type: application/json" \
  -d '{"layer_sizes": [784, 30, 10]}'
```

**List Networks:**
```bash
curl http://localhost:8000/api/networks
```

**Train a Network** (replace `{network_id}`):
```bash
curl -X POST http://localhost:8000/api/networks/{network_id}/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 5, "mini_batch_size": 10, "learning_rate": 3.0}'
```

**Delete a Network** (replace `{network_id}`):
```bash
curl -X DELETE http://localhost:8000/api/networks/{network_id}
```

## Using the Web Interface

Open your browser and go to:
```
http://localhost:8000
```

This will show a simple management page where you can:
- View all networks
- See their training status and accuracy
- Delete networks

## Connecting Your Angular Frontend

Your Angular frontend should use these endpoints:

1. **POST /api/networks** - Create a network
2. **POST /api/networks/{id}/train** - Train it
3. **WebSocket connection** - Listen for `training_update` events
4. **GET /api/networks/{id}/successful_example** - Get correct predictions
5. **GET /api/networks/{id}/unsuccessful_example** - Get incorrect predictions
6. **DELETE /api/networks/{id}** - Clean up

Example WebSocket connection in Angular:
```typescript
import { io } from 'socket.io-client';

const socket = io('http://localhost:8000');

socket.on('training_update', (data) => {
  console.log(`Epoch ${data.epoch}/${data.total_epochs}`);
  console.log(`Accuracy: ${data.accuracy * 100}%`);
  console.log(`Progress: ${data.progress}%`);
});
```

## What Changed?

✅ **Kept** (8 essential endpoints):
- Network creation, training, deletion
- Getting test examples (successful/unsuccessful)
- Training status and real-time updates
- Status checks

❌ **Removed** (6 unused endpoints):
- Single/batch predictions
- Network visualization
- Network statistics
- Finding misclassified examples
- Loading networks from disk

## Documentation

- **ENDPOINTS_REFERENCE.md** - Complete API documentation
- **CLEANUP_SUMMARY.md** - Detailed cleanup information
- **test_api.py** - Automated test script

## Troubleshooting

**Port already in use:**
```bash
pkill -f 'python src/api_server.py'
# Then restart the server
```

**Import errors:**
Make sure you're running from the project root and have all dependencies:
```bash
pip install -r requirements.txt
```

**MNIST data not found:**
The data should be in `data/mnist.pkl.gz`. If missing, the loader will try to download it.

## Next Steps

1. Start the API server
2. Run the test script to verify everything works
3. Connect your Angular frontend
4. Test the full workflow: create → train → view examples → delete

Enjoy your cleaned up neural network API! 🎉

