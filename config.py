"""
Configuration for FaceNet + MTCNN face recognition system
"""

# Paths
FACENET_MODEL_PATH = "models/facenet_pytorch.pth"  # No need to load - using pretrained
DATABASE_PATH = "output/face_database.pkl"
DATASET_PATH = "dataset"

# FaceNet parameters
FACE_SIZE = (160, 160)  # Input size for FaceNet
EMBEDDING_SIZE = 512     # Dimension of embedding vector (PyTorch: 512D)

# Recognition parameters
RECOGNITION_THRESHOLD = 0.6  # Cosine distance threshold (0.5 - 0.7 is good for 512D)
DISTANCE_METRIC = "cosine"   # cosine or euclidean

# MTCNN parameters
MIN_FACE_SIZE = 20  # Minimum face size (pixels)

# Camera
CAMERA_INDEX = 0  # 0 = default webcam

# Minimum images to register user
MIN_IMAGES_PER_USER = 1  # Just need 1 image for quick test (recommend 3-5 for high accuracy)
