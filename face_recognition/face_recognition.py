"""
Face recognition module using FaceNet
"""

import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
from numpy.linalg import norm
import pickle
import os
from config import (
    FACENET_MODEL_PATH, 
    DATABASE_PATH, 
    RECOGNITION_THRESHOLD,
    EMBEDDING_SIZE
)


class FaceRecognizer:
    """Face recognition using FaceNet embeddings"""
    
    def __init__(self, model_path=FACENET_MODEL_PATH):
        """
        Initialize FaceNet model
        
        Args:
            model_path: Path to model file (not used with PyTorch)
        """
        print("🔄 Loading FaceNet model (PyTorch)...")
        
        # Sử dụng pretrained InceptionResnetV1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        print(f"✅ FaceNet model đã sẵn sàng (Output: 512D) - Device: {self.device}")
        
        self.database = {}
        self.load_database()
    
    def get_embedding(self, face):
        """
        Convert face to embedding vector
        
        Args:
            face: Face image (160x160x3)
            
        Returns:
            Embedding vector
        """
        # Convert BGR to RGB and normalize
        face = face[:, :, ::-1]  # BGR to RGB
        face = face.astype('float32') / 255.0
        
        # Normalize according to ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face = (face - mean) / std
        
        # Convert to tensor (C, H, W)
        face = torch.from_numpy(face.transpose(2, 0, 1)).float()
        face = face.unsqueeze(0).to(self.device)
        
        # Predict embedding
        with torch.no_grad():
            embedding = self.model(face).cpu().numpy()[0]
        
        return embedding
    
    def cosine_distance(self, embedding1, embedding2):
        """
        Calculate cosine distance between 2 embeddings
        
        Args:
            embedding1, embedding2: Embedding vectors
            
        Returns:
            Cosine distance (0 = identical, 1 = completely different)
        """
        return 1 - np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    
    def euclidean_distance(self, embedding1, embedding2):
        """
        Calculate Euclidean distance between 2 embeddings
        
        Args:
            embedding1, embedding2: Embedding vectors
            
        Returns:
            Euclidean distance
        """
        return norm(embedding1 - embedding2)
    
    def recognize(self, face, threshold=RECOGNITION_THRESHOLD):
        """
        Recognize face from face image or embedding
        
        Args:
            face: Face image (160x160x3) or embedding vector (512,)
            threshold: Threshold to determine "Unknown"
            
        Returns:
            (user name, confidence 0-1)
        """
        if len(self.database) == 0:
            return "Unknown", 0.0
        
        # Check if input is image or embedding
        if len(face.shape) == 3:
            # Is image → convert to embedding
            face_embedding = self.get_embedding(face)
        else:
            # Already embedding
            face_embedding = face
        
        min_dist = float('inf')
        identity = "Unknown"
        
        # Compare with all users in database
        for name, db_embedding in self.database.items():
            dist = self.cosine_distance(face_embedding, db_embedding)
            
            if dist < min_dist:
                min_dist = dist
                identity = name
        
        # If distance > threshold → Unknown
        if min_dist > threshold:
            identity = "Unknown"
            confidence = 0.0
        else:
            # Convert distance to confidence (1 - distance)
            confidence = max(0.0, 1.0 - min_dist)
        
        return identity, confidence
    
    def add_user(self, name, embeddings):
        """
        Add new user to database
        
        Args:
            name: User name
            embeddings: List of embeddings (from multiple images)
        """
        # Get average embedding
        avg_embedding = np.mean(embeddings, axis=0)
        self.database[name] = avg_embedding
        
        print(f"✅ Added user: {name} ({len(embeddings)} images)")
    
    def save_database(self, path=DATABASE_PATH):
        """
        Save database to file
        
        Args:
            path: File path to save
        """
        with open(path, 'wb') as f:
            pickle.dump(self.database, f)
        
        print(f"💾 Saved database: {len(self.database)} users → {path}")
    
    def load_database(self, path=DATABASE_PATH):
        """
        Load database from file
        
        Args:
            path: Database file path
        """
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.database = pickle.load(f)
            print(f"📂 Loaded database: {len(self.database)} users")
        else:
            print("⚠️ No database yet, need to register users first")
            self.database = {}
    
    def get_users(self):
        """Return list of user names"""
        return list(self.database.keys())
    
    def delete_user(self, name):
        """
        Delete user from database
        
        Args:
            name: User name to delete
        """
        if name in self.database:
            del self.database[name]
            print(f"🗑️ Deleted user: {name}")
            return True
        else:
            print(f"❌ User not found: {name}")
            return False
    
    def clear_database(self):
        """Delete entire database"""
        self.database = {}
        print("🗑️ Deleted entire database")


# Test function
if __name__ == "__main__":
    print("🧪 Testing FaceNet Recognition...")
    
    try:
        recognizer = FaceRecognizer()
        
        print(f"\n📊 Statistics:")
        print(f"  - Number of users: {len(recognizer.database)}")
        print(f"  - List: {recognizer.get_users()}")
        print(f"  - Threshold: {RECOGNITION_THRESHOLD}")
        print(f"  - Device: {recognizer.device}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
