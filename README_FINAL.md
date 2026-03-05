# 📦 Dataset Manager - Lưu Vector Khuôn Mặt & Biển Số Xe

## ✅ Setup Hoàn Tất!

Tất cả file cần thiết đã được tạo và test thành công.

---

## 🚀 Bước 1: Chạy Demo (5 phút)

```bash
python demo_dataset.py
```

Chọn **option 3** để xem database structure.

---

## 🎥 Bước 2: Chạy Camera Test (10 phút)

```bash
python test_dual_cameras_with_dataset.py
```

- ✅ Tự động phát hiện khuôn mặt → lưu embedding vectors
- ✅ Tự động phát hiện biển số → lưu ảnh biển số
- ✅ Hiển thị status trực tiếp trên camera
- Nhấn **'r'** để xem báo cáo
- Nhấn **'q'** để thoát

---

## 📊 Bước 3: Xem Kết Quả

```bash
python -c "from dataset_manager import DatasetManager; m = DatasetManager(); print(m.get_summary())"
```

---

## 📚 Tài Liệu

| File | Nội Dung | Thời Gian |
|------|---------|----------|
| **SUMMARY.txt** | Tóm tắt hoàn chỉnh | 5 min |
| **QUICK_START.txt** | Hướng dẫn nhanh | 5 min |
| **README_DATASET.py** | Các tính năng & ví dụ | 10 min |
| **DATASET_USAGE_GUIDE.py** | Hướng dẫn chi tiết (CHÍNH) | 30 min |
| **FILE_INDEX.md** | Chỉ mục tất cả file | 10 min |

---

## 💻 Cách Sử Dụng (Code)

### Khởi Tạo

```python
from dataset_manager import DatasetManager
manager = DatasetManager()
```

### Lưu Khuôn Mặt

```python
manager.save_face_vector(
    name="Anh Dat",
    face_image=face_img,          # numpy array (160,160,3)
    embedding_vector=embedding,   # numpy array (512,)
    metadata={'camera': 0}
)
```

### Lưu Biển Số

```python
manager.save_license_plate(
    plate_text="29S-12345",
    plate_image=lp_img,
    metadata={'camera': 1}
)
```

### Lấy Thống Kê

```python
summary = manager.get_summary()
print(f"Persons: {summary['faces']['total_persons']}")
print(f"Vectors: {summary['faces']['total_vectors']}")
```

### Xuất Báo Cáo

```python
manager.export_face_report()
manager.export_lp_csv()
```

---

## 📁 Dữ Liệu Lưu Ở Đâu?

```
dataset/
├── face_images/{name}/              ← Ảnh khuôn mặt
└── license_plate_images/{plate}/    ← Ảnh biển số

output/
├── face_vectors/
│   ├── face_database.json           ← Metadata
│   ├── vectors.pkl                  ← Vectors (512D)
│   └── report_*.json                ← Báo cáo
│
└── license_plates/
    ├── license_plates_database.json
    ├── report_*.json
    └── license_plates_*.csv
```

---

## ✨ Các Tính Năng

✅ **Auto-save Embeddings** - Tự động lưu 512D vectors từ camera  
✅ **Auto-save License Plates** - Tự động lưu ảnh biển số  
✅ **Duplicate Prevention** - Tránh lưu trùng (configurable)  
✅ **Multiple Formats** - JSON + Binary storage  
✅ **CSV Export** - Xuất để analysis  
✅ **Real-time Display** - Hiển thị status trên camera  
✅ **Full Documentation** - 5 files hướng dẫn hoàn chỉnh  

---

## 🔧 Configuration

Trong `test_dual_cameras_with_dataset.py`:

```python
# Thay đổi save interval (tránh trùng)
self.save_interval_frames = 30    # Default: 30 frames

# Thay đổi confidence threshold (chất lượng)
self.min_confidence = 0.6         # Default: 0.6
```

---

## 📊 Files Tạo Được

✅ **dataset_manager.py** - Core module (chính)  
✅ **test_dual_cameras_with_dataset.py** - Tích hợp camera  
✅ **demo_dataset.py** - Demo interactve  
✅ **DATASET_USAGE_GUIDE.py** - Hướng dẫn chi tiết  
✅ **FILE_INDEX.md** - Chỉ mục file  
✅ **QUICK_START.txt** - Hướng dẫn nhanh  
✅ **README_DATASET.py** - Tính năng & ví dụ  
✅ **SETUP_COMPLETE.txt** - Setup info  
✅ **SUMMARY.txt** - Tóm tắt toàn bộ  

---

## ❓ FAQ

**Q: Tôi cần cài gì thêm?**  
A: Không cần! Tất cả module đã tích hợp sẵn.

**Q: Vector được lưu ở định dạng nào?**  
A: 512D floating-point array từ FaceNet.

**Q: Có thể dùng vectors để training?**  
A: Có! `manager.get_all_face_vectors()` trả về numpy array.

**Q: Làm sao để tránh lưu trùng?**  
A: Set `self.save_interval_frames = 60` để save mỗi 60 frames.

**Q: Database sẽ chiếm bao nhiêu dung lượng?**  
A: ~2KB/vector, vậy 1000 vectors = 2MB.

---

## 🎯 Next Steps

1. ✅ **Read**: SUMMARY.txt (overview)
2. ✅ **Run**: `python demo_dataset.py`
3. ✅ **Run**: `python test_dual_cameras_with_dataset.py`
4. ✅ **Learn**: DATASET_USAGE_GUIDE.py
5. ✅ **Integrate**: Vào project của bạn

---

## 📞 Support

Xem tài liệu:
- Hướng dẫn chi tiết: **DATASET_USAGE_GUIDE.py**
- Quick reference: **QUICK_START.txt**
- API reference: **FILE_INDEX.md**

---

**Status**: ✅ Ready to use!  
**Created**: 2026-01-28  
**Version**: 1.0
