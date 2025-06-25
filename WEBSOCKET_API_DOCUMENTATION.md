# WebSocket API Documentation for Neural Network Training

This document describes how to integrate with the real-time WebSocket API for neural network training progress updates. The API uses Socket.IO for WebSocket communication.

## Overview

The neural network training API provides real-time updates during training via WebSocket connections. This allows frontend applications to display live progress bars, accuracy metrics, and training status without polling.

## API Endpoint

- **Base URL**: `http://localhost:8000` (development) or your deployed server URL
- **WebSocket Protocol**: Socket.IO v4+
- **CORS**: Enabled for all origins (`*`)

## Socket.IO Events

### Client → Server Events

Currently, no client-to-server events are required. The connection is primarily server-to-client for broadcasting training updates.

### Server → Client Events

#### `training_update`

Emitted after each training epoch completes.

**Event Data Structure:**
```typescript
interface TrainingUpdate {
  job_id: string;           // Unique identifier for the training job
  network_id: string;       // Unique identifier for the neural network
  epoch: number;            // Current epoch number (1-based)
  total_epochs: number;     // Total number of epochs for this training session
  accuracy: number | null;  // Current accuracy (0.0 to 1.0), null if no test data
  elapsed_time: number;     // Time taken for this epoch in seconds
  progress: number;         // Training progress percentage (0-100)
  correct?: number;         // Number of correct predictions (if test data available)
  total?: number;          // Total number of test samples (if test data available)
}
```

**Example Event Data:**
```json
{
  "job_id": "a82c53d7-e99d-470d-ae1e-8575e42e1926",
  "network_id": "0d8da319-2268-48f3-b4bf-4a4432815aef",
  "epoch": 1,
  "total_epochs": 10,
  "accuracy": 0.8369,
  "elapsed_time": 2.65,
  "progress": 10.0,
  "correct": 8369,
  "total": 10000
}
```

## Angular Integration

### 1. Install Socket.IO Client

```bash
npm install socket.io-client
npm install --save-dev @types/socket.io-client
```

### 2. Service Implementation

Create a service to handle WebSocket connections:

```typescript
// training-websocket.service.ts
import { Injectable } from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { Observable, BehaviorSubject } from 'rxjs';

export interface TrainingUpdate {
  job_id: string;
  network_id: string;
  epoch: number;
  total_epochs: number;
  accuracy: number | null;
  elapsed_time: number;
  progress: number;
  correct?: number;
  total?: number;
}

export interface ConnectionStatus {
  connected: boolean;
  socketId?: string;
}

@Injectable({
  providedIn: 'root'
})
export class TrainingWebSocketService {
  private socket: Socket;
  private connectionStatus = new BehaviorSubject<ConnectionStatus>({ connected: false });
  private trainingUpdates = new BehaviorSubject<TrainingUpdate | null>(null);

  constructor() {
    this.initializeConnection();
  }

  private initializeConnection(): void {
    // Replace with your server URL
    const serverUrl = 'http://localhost:8000';
    
    this.socket = io(serverUrl, {
      transports: ['websocket', 'polling'], // Fallback to polling if WebSocket fails
      timeout: 20000,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });

    // Connection event handlers
    this.socket.on('connect', () => {
      console.log('Connected to training WebSocket:', this.socket.id);
      this.connectionStatus.next({ 
        connected: true, 
        socketId: this.socket.id 
      });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Disconnected from training WebSocket:', reason);
      this.connectionStatus.next({ connected: false });
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.connectionStatus.next({ connected: false });
    });

    // Training update handler
    this.socket.on('training_update', (data: TrainingUpdate) => {
      console.log('Training update received:', data);
      this.trainingUpdates.next(data);
    });
  }

  // Observable for connection status
  getConnectionStatus(): Observable<ConnectionStatus> {
    return this.connectionStatus.asObservable();
  }

  // Observable for training updates
  getTrainingUpdates(): Observable<TrainingUpdate | null> {
    return this.trainingUpdates.asObservable();
  }

  // Check if currently connected
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  // Manually reconnect
  reconnect(): void {
    if (this.socket) {
      this.socket.connect();
    }
  }

  // Disconnect
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
    }
  }

  // Cleanup on service destroy
  ngOnDestroy(): void {
    this.disconnect();
  }
}
```

### 3. Component Implementation

Use the service in your components:

```typescript
// training-progress.component.ts
import { Component, OnInit, OnDestroy } from '@angular/core';
import { Subscription } from 'rxjs';
import { TrainingWebSocketService, TrainingUpdate, ConnectionStatus } from './training-websocket.service';

@Component({
  selector: 'app-training-progress',
  template: `
    <div class="training-progress-container">
      <!-- Connection Status -->
      <div class="connection-status" [class.connected]="connectionStatus.connected">
        <span *ngIf="connectionStatus.connected; else disconnected">
          🟢 Connected (ID: {{connectionStatus.socketId}})
        </span>
        <ng-template #disconnected>
          🔴 Disconnected
          <button (click)="reconnect()" class="btn-reconnect">Reconnect</button>
        </ng-template>
      </div>

      <!-- Training Progress -->
      <div *ngIf="currentTraining" class="training-info">
        <h3>Training Progress</h3>
        
        <!-- Progress Bar -->
        <div class="progress-bar-container">
          <div class="progress-bar" [style.width.%]="currentTraining.progress">
            {{currentTraining.progress.toFixed(1)}}%
          </div>
        </div>

        <!-- Training Details -->
        <div class="training-details">
          <div class="detail-row">
            <span class="label">Epoch:</span>
            <span class="value">{{currentTraining.epoch}} / {{currentTraining.total_epochs}}</span>
          </div>
          
          <div class="detail-row" *ngIf="currentTraining.accuracy !== null">
            <span class="label">Accuracy:</span>
            <span class="value">{{(currentTraining.accuracy * 100).toFixed(2)}}%</span>
          </div>
          
          <div class="detail-row">
            <span class="label">Elapsed Time:</span>
            <span class="value">{{currentTraining.elapsed_time.toFixed(2)}}s</span>
          </div>
          
          <div class="detail-row" *ngIf="currentTraining.correct && currentTraining.total">
            <span class="label">Correct/Total:</span>
            <span class="value">{{currentTraining.correct}} / {{currentTraining.total}}</span>
          </div>
          
          <div class="detail-row">
            <span class="label">Remaining Epochs:</span>
            <span class="value">{{currentTraining.total_epochs - currentTraining.epoch}}</span>
          </div>
        </div>

        <!-- Network & Job IDs -->
        <div class="metadata">
          <small>
            <div>Job ID: {{currentTraining.job_id}}</div>
            <div>Network ID: {{currentTraining.network_id}}</div>
          </small>
        </div>
      </div>

      <!-- No Training Active -->
      <div *ngIf="!currentTraining && connectionStatus.connected" class="no-training">
        <p>No active training session. Start training a network to see real-time updates.</p>
      </div>
    </div>
  `,
  styles: [`
    .training-progress-container {
      padding: 20px;
      border: 1px solid #ddd;
      border-radius: 8px;
      margin: 20px 0;
    }

    .connection-status {
      padding: 10px;
      border-radius: 4px;
      margin-bottom: 20px;
      background-color: #ffebee;
    }

    .connection-status.connected {
      background-color: #e8f5e8;
    }

    .btn-reconnect {
      margin-left: 10px;
      padding: 4px 8px;
      background-color: #2196f3;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }

    .progress-bar-container {
      width: 100%;
      height: 30px;
      background-color: #f0f0f0;
      border-radius: 15px;
      overflow: hidden;
      margin-bottom: 20px;
    }

    .progress-bar {
      height: 100%;
      background-color: #4caf50;
      transition: width 0.5s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: bold;
    }

    .training-details {
      margin-bottom: 20px;
    }

    .detail-row {
      display: flex;
      justify-content: space-between;
      padding: 5px 0;
      border-bottom: 1px solid #eee;
    }

    .label {
      font-weight: bold;
    }

    .metadata {
      color: #666;
      font-size: 12px;
      border-top: 1px solid #eee;
      padding-top: 10px;
    }

    .no-training {
      text-align: center;
      color: #666;
      font-style: italic;
    }
  `]
})
export class TrainingProgressComponent implements OnInit, OnDestroy {
  connectionStatus: ConnectionStatus = { connected: false };
  currentTraining: TrainingUpdate | null = null;
  
  private subscriptions: Subscription[] = [];

  constructor(private trainingWebSocket: TrainingWebSocketService) {}

  ngOnInit(): void {
    // Subscribe to connection status
    this.subscriptions.push(
      this.trainingWebSocket.getConnectionStatus().subscribe(
        status => this.connectionStatus = status
      )
    );

    // Subscribe to training updates
    this.subscriptions.push(
      this.trainingWebSocket.getTrainingUpdates().subscribe(
        update => this.currentTraining = update
      )
    );
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  reconnect(): void {
    this.trainingWebSocket.reconnect();
  }
}
```

### 4. Module Configuration

Add the service to your module:

```typescript
// app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { TrainingWebSocketService } from './training-websocket.service';
import { TrainingProgressComponent } from './training-progress.component';

@NgModule({
  declarations: [
    TrainingProgressComponent
  ],
  imports: [
    BrowserModule
  ],
  providers: [
    TrainingWebSocketService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
```

## Usage Workflow

1. **Establish Connection**: The WebSocket service automatically connects when instantiated
2. **Start Training**: Use the REST API to start training a neural network
3. **Receive Updates**: Your Angular app will automatically receive `training_update` events
4. **Display Progress**: Update your UI with real-time training progress

### Starting Training (REST API)

```typescript
// Use Angular HttpClient to start training
startTraining(networkId: string, epochs: number = 10): Observable<any> {
  const trainingParams = {
    epochs: epochs,
    mini_batch_size: 10,
    learning_rate: 3.0
  };

  return this.http.post(
    `http://localhost:8000/api/networks/${networkId}/train`,
    trainingParams
  );
}
```

## Error Handling

The service includes built-in error handling for:
- Connection failures
- Reconnection attempts
- Network interruptions
- Server disconnections

## Best Practices

1. **Connection Management**: Always check connection status before relying on real-time updates
2. **Reconnection**: Implement automatic reconnection with exponential backoff
3. **Memory Management**: Unsubscribe from observables in component `ngOnDestroy`
4. **Error Handling**: Display user-friendly messages for connection issues
5. **Fallback**: Consider polling the training status endpoint as a fallback when WebSocket is unavailable

## Testing

You can test the WebSocket connection using the provided test page at:
`http://localhost:8000/test_socketio.html`

This will help verify that the WebSocket server is working correctly before integrating with your Angular application.

## Troubleshooting

### Common Issues:

1. **CORS Errors**: Ensure your server URL is correct and CORS is properly configured
2. **Connection Refused**: Verify the server is running and accessible
3. **No Updates Received**: Check that training is actually running and the network exists
4. **Polling Fallback**: If WebSocket fails, Socket.IO will automatically fall back to HTTP polling

### Debug Tips:

- Enable Socket.IO debug logging: `localStorage.debug = 'socket.io-client:socket';`
- Check browser network tab for WebSocket connections
- Monitor server logs for emitted events
- Use the test HTML page to verify server-side functionality
