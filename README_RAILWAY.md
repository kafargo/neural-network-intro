# Deploying to Railway

This README contains instructions for deploying the neural network application to Railway.

## Prerequisites

1. A Railway account (https://railway.app)
2. Railway CLI installed locally (optional, for local deployment)

## Deployment Steps

### Option 1: Using Railway CLI

1. Install Railway CLI if not already installed:

   ```
   npm install -g @railway/cli
   ```

2. Login to Railway:

   ```
   railway login
   ```

3. Initialize a new project:

   ```
   railway init
   ```

4. Deploy the application:
   ```
   railway up
   ```

### Option 2: Using Railway Dashboard

1. Fork or clone this repository to your GitHub account.
2. Go to Railway dashboard and click "New Project".
3. Select "Deploy from GitHub repo".
4. Select your forked/cloned repository.
5. Railway will detect the Dockerfile and build and deploy automatically.

## Project Structure for Containerization

- `Dockerfile`: Main container for the Flask application
- `Dockerfile.cron`: Container for the cron job that cleans up models daily
- `docker-compose.yml`: Compose file for local development with both services
- `cleanup_models.sh`: Script that removes all saved models (runs daily in production)
- `railway.json`: Configuration for Railway deployment

## Environment Variables

None required for basic operation.

## Notes on Model Persistence

- In the Railway deployment, models are stored in a volume and cleaned up daily at midnight UTC.
- The cleanup happens via a cron job running in a separate container.

## Local Testing with Docker

To test the containerized setup locally before deploying:

```bash
docker-compose up
```

This will start both the application and the cron service.
