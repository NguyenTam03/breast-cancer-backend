# BreastCare AI Backend

FastAPI backend for BreastCare AI - AI-powered breast cancer diagnosis mobile application.

## Tech Stack

- **FastAPI** - Modern Python web framework
- **MongoDB** - NoSQL database with Beanie ODM
- **TensorFlow** - Machine learning framework
- **JWT** - Authentication
- **Docker** - Containerization

## Quick Start

### Prerequisites

- Python 3.11+
- MongoDB 6.0+
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment file:
```bash
cp .env.example .env
```

5. Update `.env` with your settings:
```
MONGODB_URL=mongodb://localhost:27017/breastcare
JWT_SECRET_KEY=your-super-secret-jwt-key-here
```

### Running the Application

#### Development Mode

```bash
# From backend directory
python -m app.main
```

or

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Using Docker

```bash
# Build and run with docker-compose
docker-compose up --build
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── core/
│   │   ├── config.py        # Configuration settings
│   │   └── database.py      # MongoDB connection
│   ├── models/              # Beanie document models
│   │   ├── user.py
│   │   ├── analysis.py
│   │   └── auth.py
│   ├── api/                 # API routes
│   │   ├── auth.py
│   │   ├── users.py
│   │   ├── analysis.py
│   │   └── health.py
│   ├── services/            # Business logic
│   ├── ml/                  # ML inference
│   └── utils/               # Utilities
├── models/                  # Trained ML models
├── uploads/                 # Uploaded images
├── tests/                   # Test files
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## API Endpoints

### Health Check
- `GET /api/v1/health/` - Basic health check
- `GET /api/v1/health/detailed` - Detailed system info
- `GET /api/v1/health/status` - Service status

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh access token
- `POST /api/v1/auth/logout` - User logout

### Users
- `GET /api/v1/users/profile` - Get user profile
- `PUT /api/v1/users/profile` - Update user profile

### Analysis
- `POST /api/v1/analysis/predict` - Analyze image
- `GET /api/v1/analysis/history` - Get analysis history
- `GET /api/v1/analysis/{id}` - Get analysis details

## Development

### Running Tests

```bash
pytest
```

### Code Style

```bash
# Format code
black app/
isort app/

# Lint code
flake8 app/
```

### Database Migration

The application automatically creates indexes on startup. For manual database operations:

```python
from app.core.database import init_database
import asyncio

asyncio.run(init_database())
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URL` | MongoDB connection string | `mongodb://localhost:27017/breastcare` |
| `JWT_SECRET_KEY` | JWT secret key | Required |
| `DEBUG` | Debug mode | `True` |
| `MAX_UPLOAD_SIZE` | Max file upload size (bytes) | `10485760` (10MB) |

## ML Models

The application expects the following ML models in the `models/` directory:

- `feature_extractor.h5` - CNN feature extractor
- `model_gwo_selected_feature.h5` - GWO optimized model
- `gwo_feature_indices.npy` - Selected feature indices

## Docker

### Build Image

```bash
docker build -t breastcare-api .
```

### Run Container

```bash
docker run -p 8000:8000 -e MONGODB_URL=mongodb://host.docker.internal:27017/breastcare breastcare-api
```

## License

MIT License
