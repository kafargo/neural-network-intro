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

#### Delete All Networks

```
DELETE /api/networks
```

Delete all networks (from both memory and disk).

**Response**

```json
{
  "deleted_count": 5,
  "deleted_from_memory": 3,
  "deleted_from_disk": 5,
  "message": "Successfully deleted 5 network(s)"
}
```

**Note**: This is a destructive operation that cannot be undone. All trained networks will be permanently deleted.


### Training

#### Train Network

```
POST /api/networks/{network_id}/train
```

Start asynchronous training for the specified network.

**Request Body**

```json
{
  "epochs": 5,
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
  "epochs": 5,
  "accuracy": 0.946
}
```

Possible status values: `pending`, `training`, `completed`, `failed`

If status is `failed`, an additional `error` field will be included with the error message.

**Note**: For real-time training progress updates, use the WebSocket API instead of polling this endpoint. See the [WebSocket API Documentation](WEBSOCKET_API_DOCUMENTATION.md) for details.


### Visualizations


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
  "image_data": "iVBORw0KGgoAAAANSUhEUgAA...(base64 encoded PNG image)",
  "output_weights": [[0.1, -0.2, ...], ...],
  "network_output": [0.01, 0.02, 0.01, 0.03, 0.02, 0.85, 0.01, 0.02, 0.01, 0.02]
}
```

**Note**: `image_data` is a base64-encoded PNG image (without the `data:image/png;base64,` prefix). To use in HTML, prepend `data:image/png;base64,` to the value.

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
  "image_data": "iVBORw0KGgoAAAANSUhEUgAA...(base64 encoded PNG image)",
  "output_weights": [[0.1, -0.2, ...], ...],
  "network_output": [0.05, 0.15, 0.01, 0.03, 0.02, 0.05, 0.01, 0.62, 0.01, 0.05]
}
```

**Note**: `image_data` is a base64-encoded PNG image (without the `data:image/png;base64,` prefix). To use in HTML, prepend `data:image/png;base64,` to the value.

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
    epochs: number = 5,
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

  // Delete a network
  deleteNetwork(networkId: string): Observable<any> {
    return this.http.delete(`${this.apiUrl}/networks/${networkId}`);
  }

  // Delete all networks
  deleteAllNetworks(): Observable<any> {
    return this.http.delete(`${this.apiUrl}/networks`);
  }

  // Get a successful example
  getSuccessfulExample(networkId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/networks/${networkId}/successful_example`);
  }

  // Get an unsuccessful example
  getUnsuccessfulExample(networkId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/networks/${networkId}/unsuccessful_example`);
  }
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
