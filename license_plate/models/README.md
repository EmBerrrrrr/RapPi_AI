# License Plate Detection Models

Please place the following model files in this directory:

## Required Files:
1. **LP_best.pt** - YOLOv8 License Plate Detection Model
   - File này để phát hiện vị trí biển số xe trong ảnh
   
2. **Letters_detection.pt** - YOLOv8 OCR Model  
   - File này để đọc các ký tự trên biển số

## Download Models:
These models are not included in the repository due to file size.

### Option 1: Train Your Own
Follow the instructions in the Vietnamese License Plate Detection repository to train custom models.

### Option 2: Use Pretrained
Contact the repository maintainer for pretrained models.

## File Structure:
```
license_plate/
└── models/
    ├── LP_best.pt           (YOLOv8 LP detector)
    ├── Letters_detection.pt (YOLOv8 OCR)
    └── README.md           (this file)
```

## Model Info:
- Framework: YOLOv8 (Ultralytics)
- Input: RGB images
- Output: Bounding boxes + OCR text
- Supported: Vietnamese license plates

---
**Note:** Without these models, the license plate detection feature will not work.
