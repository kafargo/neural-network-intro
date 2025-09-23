# Neural Network API Documentation

This document provides comprehensive documentation for the Neural Network API endpoints. It can be used by front-end applications (like Angular) to make calls to the backend.

## Base URL

In development: `http://localhost:8000`  
In production: `https://neural-network-intro-production.up.railway.app/`

## Authentication

Currently, the API doesn't require authentication.

## Response Format

All API responses are in JSON format. Successful responses typically include relevant data and a status code of 200. Error responses include an `error` field with a description and an appropriate HTTP status code.

## API Endpoints

### Status

#### Get API Status

```
GET /api/status
```

Check if the API is running and get basic information about active networks and training jobs.

**Response**

```json
{
  "status": "online",
  "active_networks": 2,
  "training_jobs": 0
}
```

### Networks

#### Create Network

```
POST /api/networks
```

Create a new neural network with the specified architecture.

**Request Body**

```json
{
  "layer_sizes": [784, 30, 10]
}
```

`layer_sizes` is optional and defaults to `[784, 30, 10]` (standard for MNIST digit recognition).

**Response**

```json
{
  "network_id": "550e8400-e29b-41d4-a716-446655440000",
  "architecture": [784, 30, 10],
  "status": "created"
}
```

#### List Networks

```
GET /api/networks
```

List all available networks, both in-memory and saved.

**Response**

```json
{
  "networks": [
    {
      "network_id": "550e8400-e29b-41d4-a716-446655440000",
      "architecture": [784, 30, 10],
      "trained": true,
      "accuracy": 0.946,
      "status": "in_memory"
    },
    {
      "network_id": "550e8400-e29b-41d4-a716-446655440001",
      "architecture": [784, 50, 10],
      "trained": true,
      "accuracy": 0.952,
      "status": "saved"
    }
  ]
}
```

#### Load Network

```
POST /api/networks/{network_id}/load
```

Load a previously saved network into memory.

**Response**

```json
{
  "network_id": "550e8400-e29b-41d4-a716-446655440000",
  "architecture": [784, 30, 10],
  "status": "loaded"
}
```

#### Delete Network

```
DELETE /api/networks/{network_id}
```

Delete a network (from both memory and disk if it exists).

**Response**

```json
{
  "network_id": "550e8400-e29b-41d4-a716-446655440000",
  "deleted_from_memory": true,
  "deleted_from_disk": true
}
```

#### Get Network Statistics

```
GET /api/networks/{network_id}/stats
```

Get statistical information about a network, including accuracy and weight/bias statistics.

**Response**

```json
{
  "network_id": "550e8400-e29b-41d4-a716-446655440000",
  "architecture": [784, 30, 10],
  "trained": true,
  "accuracy": 0.946,
  "weight_stats": [
    {
      "layer": 1,
      "mean": 0.0012,
      "min": -2.3456,
      "max": 2.1234,
      "std": 0.5678,
      "shape": [30, 784]
    },
    {
      "layer": 2,
      "mean": -0.0023,
      "min": -1.5678,
      "max": 1.6789,
      "std": 0.4567,
      "shape": [10, 30]
    }
  ],
  "bias_stats": [
    {
      "layer": 1,
      "mean": 0.1234,
      "min": -0.9876,
      "max": 1.2345,
      "std": 0.3456,
      "shape": [30, 1]
    },
    {
      "layer": 2,
      "mean": 0.2345,
      "min": -0.8765,
      "max": 0.9876,
      "std": 0.3456,
      "shape": [10, 1]
    }
  ]
}
```

### Training

#### Train Network

```
POST /api/networks/{network_id}/train
```

Start asynchronous training for the specified network.

**Request Body**

```json
{
  "epochs": 10,
  "mini_batch_size": 10,
  "learning_rate": 3.0
}
```

All parameters are optional with the defaults shown above.

**Response**

```json
{
  "job_id": "660e8400-e29b-41d4-a716-446655440000",
  "network_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "training_started"
}
```

#### Get Training Status

```
GET /api/training/{job_id}
```

Get the status of a training job.

**Response**

```json
{
  "network_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "epochs": 10,
  "accuracy": 0.946
}
```

Possible status values: `pending`, `completed`, `failed`

### Predictions

#### Predict Single Example

```
POST /api/networks/{network_id}/predict
```

Run prediction on a specific example from the test dataset.

**Request Body**

```json
{
  "example_index": 42
}
```

**Response**

```json
{
  "example_index": 42,
  "predicted_digit": 7,
  "actual_digit": 7,
  "confidence_scores": [
    0.01, 0.02, 0.01, 0.03, 0.02, 0.05, 0.01, 0.87, 0.01, 0.02
  ],
  "correct": true
}
```

#### Predict Batch

```
POST /api/networks/{network_id}/predict_batch
```

Run prediction on a batch of examples from the test dataset.

**Request Body**

```json
{
  "start_index": 50,
  "count": 5
}
```

Both parameters are optional. Default `start_index` is 0, default `count` is 10.

**Response**

```json
{
  "results": [
    {
      "example_index": 50,
      "predicted_digit": 9,
      "actual_digit": 9,
      "correct": true,
      "image_data": "data:image/png;base64,..."
    },
    {
      "example_index": 51,
      "predicted_digit": 2,
      "actual_digit": 2,
      "correct": true,
      "image_data": "data:image/png;base64,..."
    }
    // ... more results
  ],
  "total": 5
}
```

### Visualizations

#### Network Visualization

```
GET /api/networks/{network_id}/visualize
```

Get a visual representation of the network structure.

**Response**

```json
{
  "network_id": "550e8400-e29b-41d4-a716-446655440000",
  "visualization": "data:image/png;base64,..."
}
```

#### Get Misclassified Examples

```
GET /api/networks/{network_id}/misclassified?max_count=5&max_check=200
```

Find misclassified examples from the test dataset.

**Query Parameters**

- `max_count`: Maximum number of misclassifications to return (default: 10)
- `max_check`: Maximum number of examples to check (default: 500)

**Response**

```json
{
  "misclassified": [
    {
      "example_index": 123,
      "predicted_digit": 7,
      "actual_digit": 1,
      "image_data": "data:image/png;base64,..."
    },
    {
      "example_index": 456,
      "predicted_digit": 9,
      "actual_digit": 4,
      "image_data": "data:image/png;base64,..."
    }
    // ... more examples
  ]
}
```

#### Get Successful Example

```
GET /api/networks/{network_id}/successful_example
```

Return a random successful example prediction with network output details.

**Response**

```json
{
  "network_id": "550e8400-e29b-41d4-a716-446655440000",
  "example_index": 789,
  "predicted_digit": 5,
  "actual_digit": 5,
  "image_data": "data:image/png;base64,...",
  "output_weights": [...],
  "network_output": [0.01, 0.02, 0.01, 0.03, 0.02, 0.85, 0.01, 0.02, 0.01, 0.02]
}
```

#### Get Unsuccessful Example

```
GET /api/networks/{network_id}/unsuccessful_example
```

Return a random unsuccessful example prediction with network output details.

**Response**

```json
{
  "network_id": "550e8400-e29b-41d4-a716-446655440000",
  "example_index": 123,
  "predicted_digit": 7,
  "actual_digit": 1,
  "image_data": "data:image/png;base64,...",
  "output_weights": [...],
  "network_output": [0.05, 0.15, 0.01, 0.03, 0.02, 0.05, 0.01, 0.62, 0.01, 0.05]
}
```

## Using with Angular

To use these endpoints in an Angular application, you can create a service to handle API calls. Here's an example:

```typescript
import { Injectable } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { Observable } from "rxjs";

@Injectable({
  providedIn: "root",
})
export class NeuralNetworkService {
  private apiUrl = "http://localhost:8000/api";

  constructor(private http: HttpClient) {}

  // Get API status
  getStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/status`);
  }

  // Create a new network
  createNetwork(layerSizes: number[] = [784, 30, 10]): Observable<any> {
    return this.http.post(`${this.apiUrl}/networks`, {
      layer_sizes: layerSizes,
    });
  }

  // List all networks
  listNetworks(): Observable<any> {
    return this.http.get(`${this.apiUrl}/networks`);
  }

  // Train a network
  trainNetwork(
    networkId: string,
    epochs: number = 10,
    miniBatchSize: number = 10,
    learningRate: number = 3.0
  ): Observable<any> {
    return this.http.post(`${this.apiUrl}/networks/${networkId}/train`, {
      epochs,
      mini_batch_size: miniBatchSize,
      learning_rate: learningRate,
    });
  }

  // Get training status
  getTrainingStatus(jobId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/training/${jobId}`);
  }

  // Predict a single example
  predict(networkId: string, exampleIndex: number): Observable<any> {
    return this.http.post(`${this.apiUrl}/networks/${networkId}/predict`, {
      example_index: exampleIndex,
    });
  }

  // Get network visualization
  getVisualization(networkId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/networks/${networkId}/visualize`);
  }

  // Get misclassified examples
  getMisclassifiedExamples(
    networkId: string,
    maxCount: number = 10,
    maxCheck: number = 500
  ): Observable<any> {
    return this.http.get(
      `${this.apiUrl}/networks/${networkId}/misclassified?max_count=${maxCount}&max_check=${maxCheck}`
    );
  }

  // More methods for other endpoints...
}
```

Make sure to include `HttpClientModule` in your Angular application's imports:

```typescript
import { NgModule } from "@angular/core";
import { BrowserModule } from "@angular/platform-browser";
import { HttpClientModule } from "@angular/common/http";

import { AppComponent } from "./app.component";

@NgModule({
  declarations: [AppComponent],
  imports: [BrowserModule, HttpClientModule],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
```
