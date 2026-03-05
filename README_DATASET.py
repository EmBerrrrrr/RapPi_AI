"""
# 📚 DATASET MANAGER - README

## 📋 Giải Pháp Lưu Vector Khuôn Mặt và Biển Số Xe

### ✨ Tính năng chính:

✅ **Lưu Vector Khuôn Mặt (Face Embeddings)**
   - Lưu 512D vector từ FaceNet
   - Tổ chức theo tên người
   - Quản lý metadata (camera, location, time)

✅ **Lưu Biển Số Xe (License Plates)**
   - Lưu ảnh và metadata
   - Tracking số lần xuất hiện
   - Tổ chức theo text biển số

✅ **Database Management**
   - JSON format cho metadata
   - Pickle format cho vectors
   - CSV export cho báo cáo

✅ **Auto-save trong Camera Stream**
   - Tích hợp vào dual camera test
   - Tránh trùng lặp (configurable interval)
   - Real-time statistics display

---

## 🚀 QUICK START

### 1. Cài đặt

```bash
# Không cần cài gì thêm - dùng module có sẵn
python demo_dataset.py
```

### 2. Lưu Dataset từ Camera (2 cách)

**Cách A: Chạy tự động (Khuyến nghị)**
```bash
python test_dual_cameras_with_dataset.py
```
- Tự động phát hiện và lưu khuôn mặt
- Tự động phát hiện và lưu biển số
- Không cần tương tác

**Cách B: Manual control**
```python
from dataset_manager import DatasetManager
manager = DatasetManager()

# Lưu khuôn mặt
manager.save_face_vector(
    name="Anh Dat",
    face_image=face_img,
    embedding_vector=embedding
)

# Lưu biển số
manager.save_license_plate(
    plate_text="29S-12345",
    plate_image=lp_img
)
```

### 3. Xem Kết Quả

```bash
# Xem báo cáo
python -c "from dataset_manager import DatasetManager; \
m = DatasetManager(); \
print(m.get_summary())"

# Xuất CSV
python -c "from dataset_manager import DatasetManager; \
m = DatasetManager(); \
m.export_face_report(); \
m.export_lp_report()"
```

---

## 📂 CẤU TRÚC FILE

```
project/
├── dataset_manager.py              ← Main module
├── test_dual_cameras_with_dataset.py    ← Integrated demo
├── demo_dataset.py                 ← Interactive demo
├── DATASET_USAGE_GUIDE.py         ← Detailed guide
│
├── dataset/
│   ├── face_images/               ← Face images organized by name
│   │   ├── Anh_Dat/
│   │   ├── Co_Van/
│   │   └── ...
│   └── license_plate_images/      ← License plate images
│       ├── 29S_12345/
│       └── ...
│
└── output/
    ├── face_vectors/              ← Face embeddings
    │   ├── face_database.json     ← Metadata
    │   ├── vectors.pkl            ← Binary vectors
    │   └── report_*.json          ← Reports
    │
    └── license_plates/            ← License plate data
        ├── license_plates_database.json
        └── report_*.json
```

---

## 🎯 USAGE EXAMPLES

### Example 1: Lưu từ Camera Stream

```python
from dataset_manager import DatasetManager
from face_recognition import FaceRecognizer
import cv2

manager = DatasetManager()
recognizer = FaceRecognizer()
cap = cv2.VideoCapture(0)

frame_count = 0
last_save = {}
SAVE_INTERVAL = 30

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % 5 == 0:
        # Detect and recognize
        faces, boxes = detect_faces(frame)
        
        for face in faces:
            name, conf = recognizer.recognize(face)
            embedding = recognizer.get_embedding(face)
            
            # Save with interval check
            if name not in last_save or (frame_count - last_save[name]) > SAVE_INTERVAL:
                manager.save_face_vector(name, face, embedding)
                last_save[name] = frame_count

cap.release()
```

### Example 2: Lưu từ Dataset Folder

```python
from dataset_manager import DatasetManager
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
import cv2
from pathlib import Path

manager = DatasetManager()
detector = FaceDetector()
recognizer = FaceRecognizer()

# Process all images in a folder
for img_path in Path("dataset/Anh_Dat").glob("*.jpg"):
    image = cv2.imread(str(img_path))
    faces, _ = detector.extract_all_faces(image)
    
    for face in faces:
        embedding = recognizer.get_embedding(face)
        manager.save_face_vector("Anh Dat", face, embedding)

# Check results
stats = manager.get_face_vector_stats()
print(f"Saved: {stats['persons']['Anh Dat']['count']} vectors")
```

### Example 3: Xuất Báo Cáo

```python
from dataset_manager import DatasetManager

manager = DatasetManager()

# JSON report
manager.export_face_report()
manager.export_lp_report()

# CSV export
manager.export_face_vectors_csv()
manager.export_lp_csv()

# Statistics
summary = manager.get_summary()
print(f"Persons: {summary['faces']['total_persons']}")
print(f"Vectors: {summary['faces']['total_vectors']}")
print(f"Plates: {summary['license_plates']['total_unique_plates']}")
```

---

## 📊 DATA FORMAT

### Face Vectors

**face_database.json:**
```json
{
  "Anh Dat": {
    "name": "Anh Dat",
    "vectors": [
      {
        "id": "Anh Dat_20260128_093045",
        "timestamp": "20260128_093045",
        "image_path": "face_images/Anh_Dat/20260128_093045.jpg"
      }
    ],
    "created_at": "20260128_093045",
    "count": 5
  }
}
```

**vectors.pkl:**
```python
{
  "Anh Dat": [
    {
      "id": "Anh Dat_20260128_093045",
      "vector": [0.123, 0.456, ..., 0.789],  # 512D
      "timestamp": "20260128_093045"
    }
  ]
}
```

### License Plates

**license_plates_database.json:**
```json
{
  "29S-12345": {
    "plate_text": "29S-12345",
    "images": [
      {
        "timestamp": "20260128_093045",
        "image_path": "license_plate_images/29S_12345/20260128_093045.jpg"
      }
    ],
    "count": 3
  }
}
```

---

## ⚙️ CONFIGURATION

### Adjust Save Interval

```python
# In test_dual_cameras_with_dataset.py
self.save_interval_frames = 60  # Save every 60 frames instead of 30
```

### Change Directories

```python
manager = DatasetManager(
    dataset_dir="my_dataset",
    output_dir="my_output"
)
```

### Custom Metadata

```python
manager.save_face_vector(
    name="Anh Dat",
    face_image=face,
    embedding_vector=embedding,
    metadata={
        'location': 'Entrance',
        'camera_id': 0,
        'confidence': 0.95,
        'custom_field': 'value'
    }
)
```

---

## 🔍 TROUBLESHOOTING

**Q: Không lưu được vector?**
- A: Check embedding size = 512D
- A: Face image should be 160x160x3
- A: numpy array, not tensor

**Q: Không thấy file được lưu?**
- A: Check output/ và dataset/ folder tồn tại
- A: Run: `python demo_dataset.py` để verify

**Q: Vector lưu trùng lặp?**
- A: Tăng `save_interval_frames`
- A: Check frame counter không reset

**Q: Memory quá lớn?**
- A: Vectors lưu dưới dạng float32 (512 * 4 bytes = 2KB/vector)
- A: Chiếm ~2GB cho 1 triệu vectors

---

## 💡 BEST PRACTICES

1. ✅ Đặt tên người rõ ràng (không dấu, underscores)
2. ✅ Lưu metadata đầy đủ (camera, location, time)
3. ✅ Check confidence > 0.6 trước khi lưu
4. ✅ Interval save 30-60 frames để tránh trùng
5. ✅ Backup dataset hàng tuần
6. ✅ Export CSV cho tracking
7. ✅ Remove low-quality vectors định kỳ

---

## 📞 SUPPORT

**File references:**
- Main: `dataset_manager.py`
- Example: `test_dual_cameras_with_dataset.py`
- Demo: `demo_dataset.py`
- Guide: `DATASET_USAGE_GUIDE.py`

**Run demo:**
```bash
python demo_dataset.py
```

---

## 🎓 LEARNING RESOURCES

1. **Start here:** Run `demo_dataset.py` → Option 3 (View Database)
2. **Full guide:** Check `DATASET_USAGE_GUIDE.py`
3. **Integration:** Copy code from `test_dual_cameras_with_dataset.py`
4. **API docs:** Check docstrings in `dataset_manager.py`

---

Generated: 2026-01-28
Version: 1.0
"""

if __name__ == "__main__":
    print(__doc__)
