import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentiment_model.pkl')
    VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_FEATURES = 5000
    
    # App settings
    SECRET_KEY = 'your-secret-key-here'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Sentiment labels
    SENTIMENT_LABELS = {
        -1: 'Negative',
        0: 'Neutral',
        1: 'Positive'
    }