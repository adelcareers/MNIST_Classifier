# MNIST Digit Classifier

A web application that uses deep learning to classify handwritten digits with a drawing interface and prediction logging.

## Project Structure
```
MNIST_Classifier/
├── models/              # Directory for saved model weights
├── app.py              # Streamlit web application
├── train.py            # Training script
├── model.py            # Model architecture
├── inference.py        # Inference pipeline
├── db_logger.py        # Database logging utilities
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Multi-container setup
└── requirements.txt    # Python dependencies
```

## Running Locally

### Prerequisites
- Python 3.8+
- PostgreSQL
- pip

### Setup
1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up PostgreSQL database:
```bash
sudo -u postgres psql
CREATE DATABASE mnist;
```

4. Set environment variables:
```bash
export POSTGRES_HOST=localhost
export POSTGRES_DB=mnist
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
```

5. Run the application:
```bash
streamlit run app.py
```

## Running with Docker

### Prerequisites
- Docker
- Docker Compose V2

### Starting the Application
1. Build and start containers:
```bash
docker compose up --build
```

2. Access the application at `http://localhost:8501`

### Managing Containers

- Start in detached mode:
```bash
docker compose up -d
```

- Stop all containers:
```bash
docker compose down
```

- Restart web service:
```bash
docker compose restart web
```

- View logs:
```bash
docker compose logs -f web  # Web service logs
docker compose logs -f db   # Database logs
```

- Check container status:
```bash
docker compose ps
```

## Using the Application

### Web Interface (app.py)
1. Start the application:
```bash
streamlit run app.py
```

2. Access the interface at `http://localhost:8501`

3. Using the drawing interface:
   - Draw a digit using the white drawing canvas
   - Canvas settings:
     - White stroke on black background
     - 280x280 pixels drawing area
     - Adjustable stroke width (default: 20)

4. Making predictions:
   - Click "Predict" button
   - View results:
     - Predicted digit
     - Confidence score
   - Optionally enter the true digit if the prediction was incorrect
   - All predictions are automatically saved to the database for logging

### Command Line Inference (inference.py)
Test the model directly with image files:

```bash
python inference.py
```

Image requirements:
- Grayscale or RGB format (will be converted to grayscale)
- Common formats (PNG, JPEG)
- Will be automatically resized to 28x28 pixels

To use your own test image:
1. Place the image in the project directory
2. Update the image path in inference.py:
```python
image_path = 'your_image.png'
```

## Training the Model

1. Train a new model:
```bash
python train.py
```

The trained model will be saved to `models/mnist_model.pth`

## Database Management

The application automatically:
- Initializes the database on startup
- Logs predictions and user feedback
- Maintains persistent storage through Docker volumes

## Troubleshooting

1. If database connection fails:
   - Check PostgreSQL service: `sudo systemctl status postgresql`
   - Verify credentials in environment variables
   - Ensure database exists: `psql -U postgres -l`

2. If model loading fails:
   - Verify model file exists in `models/` directory
   - Check file permissions

3. If Docker containers fail:
   - Check Docker daemon: `sudo systemctl status docker`
   - View detailed logs: `docker compose logs`
   - Rebuild containers: `docker compose up --build`


