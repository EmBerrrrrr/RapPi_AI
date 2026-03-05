# 🚗 Parking Management System - AI Service

AI Service cho hệ thống quản lý bãi đỗ xe thông minh với Face Recognition và License Plate Detection.

## 🌟 Tính năng

### ✅ AI Features
- **Face Detection**: Phát hiện khuôn mặt tài xế (MTCNN)
- **Face Recognition**: Nhận diện người dùng (FaceNet)  
- **License Plate Detection**: Nhận diện biển số xe Việt Nam (YOLO)
- **Auto Classification**: Tự động phân loại 3 loại khách hàng:
  - 🆕 **Khách vãng lai** (Guest)
  - 👤 **Người dùng đã đăng ký** (Registered User)
  - 🎫 **Người dùng có vé tháng** (VIP Monthly Pass)

### ✅ Business Logic
- **Check-in Flow**: Tự động nhận diện và mở barrier
- **Check-out Flow**: Tính phí và xử lý thanh toán
- **Payment Integration**: 
  - Auto-deduct từ ví
  - Thanh toán online (MoMo/VNPay)
  - Thanh toán tiền mặt
- **Loyalty Points**: Tích điểm tự động
- **Monthly Pass**: Hỗ trợ vé tháng miễn phí

## 📁 Cấu trúc Project

```
Test model/
├── face_detection.py         # Face detection với MTCNN
├── face_recognition.py       # Face recognition với FaceNet  
├── license_plate/
│   └── detector.py          # License plate detection
├── database_models.py        # PostgreSQL models (SQLAlchemy)
├── parking_service.py        # Business logic chính
├── api_server.py            # REST API (Flask)
├── config.py                # Configuration
├── quick_start.py           # Script khởi tạo
├── requirements_api.txt     # Python dependencies
├── .env.example            # Environment template
└── INTEGRATION_GUIDE.md    # Hướng dẫn tích hợp chi tiết
```

## 🚀 Quick Start

### 1. Cài đặt Dependencies

```bash
# Install Python packages
pip install -r requirements_api.txt
```

### 2. Cấu hình Database

```bash
# Copy environment template
copy .env.example .env

# Edit .env with your PostgreSQL credentials
notepad .env
```

Update `.env`:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=parking_db
DB_USER=postgres
DB_PASSWORD=your_password
```

### 3. Khởi tạo Database

```bash
# Run quick start script
python quick_start.py
```

Output:
```
🚀 PARKING MANAGEMENT SYSTEM - INITIALIZATION
======================================================================
📦 Step 1: Connecting to PostgreSQL...
✅ Database connected successfully!

📝 Step 2: Creating sample data...
✅ Created parking lot: Main Parking Lot
✅ Created user: Nguyen Van A - Plate: 51A-12345
✅ Created VIP user: Tran Thi B - Plate: 51B-99999

✅ Sample data created successfully!
```

### 4. Chạy API Server

```bash
# Start Flask server
python api_server.py
```

Server sẽ chạy tại: `http://localhost:5000`

## 🔌 API Endpoints

### Health Check
```http
GET /health
```

### Face Detection
```http
POST /api/detect/face
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

### License Plate Detection
```http
POST /api/detect/plate
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

### Check-in
```http
POST /api/parking/checkin
Content-Type: application/json

{
  "face_image": "base64_string",
  "plate_image": "base64_string",
  "device_id": "uuid",
  "lot_id": "uuid"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "uuid",
  "license_plate": "51A-12345",
  "actor_type": "registered_user",
  "user_name": "Nguyen Van A",
  "display_message": "Welcome Back!\nFee will be auto-deducted from wallet",
  "barrier_action": "OPEN"
}
```

### Check-out
```http
POST /api/parking/checkout
Content-Type: application/json

{
  "face_image": "base64_string",
  "plate_image": "base64_string",
  "device_id": "uuid"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "uuid",
  "license_plate": "51A-12345",
  "parking_fee": 15000.00,
  "payment_status": "paid",
  "can_exit": true,
  "barrier_action": "OPEN",
  "display_message": "Payment Successful!\nFee: 15,000 VND"
}
```

## 🏗️ Kiến trúc

### Shared Database Architecture

```
┌──────────────┐
│  .NET Backend│
│  (C# API)    │
└──────┬───────┘
       │
       ▼
┌─────────────────────────┐
│  PostgreSQL Database    │ ◄─────┐
│  (Shared)               │       │
└─────────────────────────┘       │
                                  │
┌──────────────┐                  │
│ Python AI    │                  │
│ Service      ├──────────────────┘
└──────────────┘
```

**Lợi ích:**
- ✅ Data consistency
- ✅ Real-time sync
- ✅ Không cần duplicate data
- ✅ Dễ maintain

## 🔧 Tích hợp với .NET

### C# Client Example

```csharp
public class AIServiceClient
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl = "http://localhost:5000";

    public async Task<CheckInResponse> ProcessCheckInAsync(
        string faceImageBase64,
        string plateImageBase64,
        Guid deviceId)
    {
        var request = new
        {
            face_image = faceImageBase64,
            plate_image = plateImageBase64,
            device_id = deviceId.ToString()
        };

        var response = await _httpClient.PostAsJsonAsync(
            $"{_baseUrl}/api/parking/checkin", 
            request
        );

        return await response.Content.ReadFromJsonAsync<CheckInResponse>();
    }
}
```

Xem chi tiết trong [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)

## 📊 Database Schema

Sử dụng chung schema với .NET Backend:
- ✅ `users` - Người dùng
- ✅ `vehicles` - Phương tiện
- ✅ `parking_lots` - Bãi đỗ xe
- ✅ `parking_sessions` - Phiên đỗ xe
- ✅ `transactions` - Giao dịch thanh toán
- ✅ `wallets` - Ví điện tử
- ✅ `loyalty_points` - Điểm thưởng
- ✅ `monthly_passes` - Vé tháng
- ✅ `iot_devices` - Thiết bị IoT
- ✅ `recognition_logs` - Log nhận diện AI

## 🧪 Testing

### Test với cURL

```bash
# Health check
curl http://localhost:5000/health

# Test plate detection
curl -X POST http://localhost:5000/api/detect/plate \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_here"}'
```

### Test với Python

```python
import requests
import base64
import cv2

# Read image
img = cv2.imread('test_plate.jpg')
_, buffer = cv2.imencode('.jpg', img)
img_base64 = base64.b64encode(buffer).decode()

# Call API
response = requests.post(
    'http://localhost:5000/api/detect/plate',
    json={'image': img_base64}
)

print(response.json())
```

## 📝 Business Flow

### FLOW I: CHECK-IN

```
1. Xe đến cổng → Camera chụp (mặt + biển số)
2. AI nhận diện biển số
3. Tra cứu database:
   - Không tìm thấy → KHÁCH VÃNG LAI
   - Tìm thấy xe đã đăng ký:
     • Có vé tháng → VIP USER  
     • Không vé tháng → REGISTERED USER
4. Tạo parking_session
5. Hiển thị message tương ứng
6. Mở barrier
```

### FLOW II: CHECK-OUT

```
1. Xe đến cổng ra → Camera chụp
2. AI nhận diện và tìm session
3. Tính phí đỗ xe:
   - VIP (vé tháng) → Miễn phí, mở barrier
   - User đăng ký:
     • Đủ tiền ví → Trừ tự động, mở barrier
     • Không đủ → Yêu cầu thanh toán online
   - Khách vãng lai:
     • Đã thanh toán online → Mở barrier
     • Chưa thanh toán → Hiển thị QR code
4. Cập nhật session và transaction
5. Mở/đóng barrier tương ứng
```

## 🔒 Security

- ✅ Database credentials trong `.env` (không commit)
- ✅ CORS configuration cho .NET
- ✅ Request validation
- ✅ Error handling

## 📈 Performance

- ⚡ Face detection: ~100ms
- ⚡ Plate detection: ~200ms
- ⚡ Total check-in: ~500ms
- 💾 Memory: ~2GB (with models loaded)

## 🐛 Troubleshooting

### "Cannot connect to database"
```bash
# Check PostgreSQL is running
# Verify credentials in .env
# Check firewall settings
```

### "Module not found"
```bash
# Reinstall dependencies
pip install --upgrade -r requirements_api.txt
```

### "Port 5000 already in use"
```python
# Change port in api_server.py
app.run(port=5001)
```

## 📚 Documentation

- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Chi tiết tích hợp .NET
- [Database Schema](#database-schema) - Cấu trúc database
- [API Reference](#api-endpoints) - Tài liệu API đầy đủ

## 🤝 Contributing

Pull requests are welcome! 

## 📄 License

MIT License

---

**Made with ❤️ for Smart Parking Management**
