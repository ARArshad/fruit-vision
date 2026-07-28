# FruitVision

An AI-based fruit classification system that takes fruit images uploaded by users and predicts the fruit category.

## Project Overview

FruitVision automates fruit identification for retailers, farmers, and consumers by combining computer vision and deep learning. Users can upload fruit images via a simple interface and receive predictions (e.g., apple, banana, orange, mango) along with confidence scores.

### Features

- Deep learning model trained on the Fruit-360 dataset
- Image preprocessing pipeline leveraging OpenCV / Pillow
- Model served through a Python backend API
- Optional database layer for logging predictions or user feedback
- Web-based frontend built with HTML, CSS, and JavaScript for interactions

## Repository Layout

```
fruitvision/
├── backend/              # Backend API service (FastAPI or Flask)
├── frontend/             # Web UI assets (HTML/CSS/JS)
├── notebooks/            # Exploratory data analysis and experiments
├── src/
│   └── fruitvision/
│       ├── data/         # Data loading and preprocessing utilities
│       ├── models/       # Model architectures and training scripts
│       └── utils/        # Shared helpers (metrics, config, etc.)
├── tests/                # Automated tests (unit/integration)
└── README.md
```

## Getting Started

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\\Scripts\\activate   # Windows
   ```

2. **Install dependencies** (will be added as the project evolves):
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset**
   - Download the Fruit-360 dataset from [https://www.kaggle.com/moltean/fruits](https://www.kaggle.com/moltean/fruits).
   - Extract the dataset into `data/raw/fruit-360` (folder structure to be defined in upcoming steps).

4. **Next Steps**
   - Implement data ingestion and preprocessing utilities.
   - Train the classification model using the Fruit-360 dataset.
   - Build the backend API to serve predictions.
   - Develop the frontend user interface.

## Requirements Summary

- **Functional**
  - Upload fruit image
  - Predict fruit category via AI model
  - Display prediction with confidence score

- **Non-Functional**
  - Accuracy ≥ 80%
  - User-friendly interface for non-technical users
  - Response time within a few seconds per image
  - AI-optimized for fast, accurate predictions

## Technology Stack

- **Backend:** Python (FastAPI or Flask), TensorFlow/Keras, OpenCV/Pillow
- **Frontend:** HTML, CSS, JavaScript
- **Database (optional):** SQLite or MySQL
- **AI:** Convolutional Neural Networks (CNNs), TensorFlow
- **Dataset:** Fruit-360

## Milestones

1. Establish environment and dataset pipeline
2. Train and evaluate baseline CNN model
3. Build API for inference
4. Implement frontend interface
5. Add logging/database integration (optional)
6. Deploy and document the system
