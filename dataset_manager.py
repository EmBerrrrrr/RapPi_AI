"""
Dataset Manager - Lưu vector khuôn mặt và biển số xe
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import pickle


class DatasetManager:
    """Quản lý dataset cho face vectors và license plates"""
    
    def __init__(self, dataset_dir="dataset", output_dir="output"):
        """
        Khởi tạo Dataset Manager
        
        Args:
            dataset_dir: Thư mục chứa dataset
            output_dir: Thư mục chứa output
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        
        # Tạo các thư mục cần thiết
        self.face_vectors_dir = self.output_dir / "face_vectors"
        self.lp_data_dir = self.output_dir / "license_plates"
        self.face_images_dir = self.dataset_dir / "face_images"
        self.lp_images_dir = self.dataset_dir / "license_plate_images"
        
        for dir_path in [self.face_vectors_dir, self.lp_data_dir, self.face_images_dir, self.lp_images_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Database files
        self.faces_db = self.face_vectors_dir / "faces_database.json"
        self.lp_db = self.lp_data_dir / "license_plates_database.json"
        self.faces_vectors_pkl = self.face_vectors_dir / "vectors.pkl"
        
        # Load existing databases
        self.faces_data = self._load_json(self.faces_db)
        self.lp_data = self._load_json(self.lp_db)
        self.vectors_data = self._load_pickle(self.faces_vectors_pkl)
        
        print(f"✅ Dataset Manager initialized")
        print(f"   📁 Face vectors: {self.face_vectors_dir}")
        print(f"   📁 License plates: {self.lp_data_dir}")
    
    def _load_json(self, filepath):
        """Load JSON file, return empty dict if not exist"""
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_json(self, data, filepath):
        """Save data to JSON file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_pickle(self, filepath):
        """Load pickle file, return empty dict if not exist"""
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_pickle(self, data, filepath):
        """Save data to pickle file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    # ============ FACE VECTOR METHODS ============
    
    def save_face_vector(self, name, face_image, embedding_vector, metadata=None):
        """
        Lưu vector khuôn mặt
        
        Args:
            name: Tên người (str)
            face_image: Ảnh khuôn mặt (numpy array)
            embedding_vector: Vector embedding (numpy array)
            metadata: Metadata bổ sung (dict)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Tạo unique ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_id = f"{name}_{timestamp}"
            
            # Tạo thư mục cho người đó
            person_dir = self.face_images_dir / name.replace(" ", "_")
            person_dir.mkdir(parents=True, exist_ok=True)
            
            # Lưu ảnh
            image_path = person_dir / f"{timestamp}.jpg"
            cv2.imwrite(str(image_path), face_image)
            
            # Lưu vector vào database (pickle)
            if name not in self.vectors_data:
                self.vectors_data[name] = []
            
            self.vectors_data[name].append({
                'id': face_id,
                'vector': embedding_vector.tolist(),  # Convert numpy to list
                'timestamp': timestamp,
                # store image path relative to dataset dir (face_images)
                'image_path': str(image_path.relative_to(self.dataset_dir))
            })
            
            # Lưu metadata vào JSON
            if name not in self.faces_data:
                self.faces_data[name] = {
                    'name': name,
                    'vectors': [],
                    'created_at': timestamp,
                    'updated_at': timestamp,
                    'count': 0
                }
            
            self.faces_data[name]['vectors'].append({
                'id': face_id,
                'timestamp': timestamp,
                'image_path': str(image_path.relative_to(self.dataset_dir)),
                'metadata': metadata or {}
            })
            self.faces_data[name]['updated_at'] = timestamp
            self.faces_data[name]['count'] = len(self.faces_data[name]['vectors'])
            
            # Save to files
            self._save_json(self.faces_data, self.faces_db)
            self._save_pickle(self.vectors_data, self.faces_vectors_pkl)
            
            print(f"✅ Đã lưu vector khuôn mặt: {name} ({face_id})")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi lưu vector khuôn mặt: {e}")
            return False
    
    def get_all_face_vectors(self):
        """
        Lấy tất cả vectors khuôn mặt
        
        Returns:
            dict: {name: [vectors]}
        """
        result = {}
        for name, data in self.vectors_data.items():
            result[name] = np.array([
                np.array(v['vector']) for v in data
            ])
        return result
    
    def get_face_vector_stats(self):
        """
        Lấy thống kê vectors khuôn mặt
        
        Returns:
            dict: Thống kê
        """
        stats = {
            'total_persons': len(self.faces_data),
            'total_vectors': sum(d['count'] for d in self.faces_data.values()),
            'persons': {}
        }
        
        for name, data in self.faces_data.items():
            stats['persons'][name] = {
                'count': data['count'],
                'created_at': data['created_at'],
                'updated_at': data['updated_at']
            }
        
        return stats
    
    # ============ LICENSE PLATE METHODS ============
    
    def save_license_plate(self, plate_text, plate_image, metadata=None):
        """
        Lưu biển số xe
        
        Args:
            plate_text: Text biển số (str)
            plate_image: Ảnh biển số (numpy array)
            metadata: Metadata bổ sung (dict)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Clean plate text
            plate_text = plate_text.strip().upper()
            
            if not plate_text or len(plate_text) < 4:
                print(f"⚠️ Biển số không hợp lệ: {plate_text}")
                return False
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Tạo thư mục cho biển số
            plate_dir = self.lp_images_dir / plate_text.replace(" ", "_")
            plate_dir.mkdir(parents=True, exist_ok=True)
            
            # Lưu ảnh
            image_path = plate_dir / f"{timestamp}.jpg"
            cv2.imwrite(str(image_path), plate_image)
            
            # Lưu metadata
            if plate_text not in self.lp_data:
                self.lp_data[plate_text] = {
                    'plate_text': plate_text,
                    'images': [],
                    'created_at': timestamp,
                    'updated_at': timestamp,
                    'count': 0
                }
            
            self.lp_data[plate_text]['images'].append({
                'timestamp': timestamp,
                'image_path': str(image_path.relative_to(self.dataset_dir)),
                'metadata': metadata or {}
            })
            self.lp_data[plate_text]['updated_at'] = timestamp
            self.lp_data[plate_text]['count'] = len(self.lp_data[plate_text]['images'])
            
            # Save to file
            self._save_json(self.lp_data, self.lp_db)
            
            print(f"✅ Đã lưu biển số: {plate_text} ({image_path.name})")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi lưu biển số: {e}")
            return False
    
    def get_license_plate_stats(self):
        """
        Lấy thống kê biển số xe
        
        Returns:
            dict: Thống kê
        """
        stats = {
            'total_unique_plates': len(self.lp_data),
            'total_images': sum(d['count'] for d in self.lp_data.values()),
            'plates': {}
        }
        
        for plate_text, data in self.lp_data.items():
            stats['plates'][plate_text] = {
                'count': data['count'],
                'created_at': data['created_at'],
                'updated_at': data['updated_at']
            }
        
        return stats
    
    # ============ EXPORT METHODS ============
    
    def export_face_report(self):
        """Xuất báo cáo faces"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_face_vector_stats(),
            'database_file': str(self.faces_db),
            'vectors_file': str(self.faces_vectors_pkl)
        }
        
        report_path = self.face_vectors_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._save_json(report, report_path)
        
        print(f"📊 Báo cáo faces: {report_path}")
        return report
    
    def export_lp_report(self):
        """Xuất báo cáo license plates"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_license_plate_stats(),
            'database_file': str(self.lp_db)
        }
        
        report_path = self.lp_data_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._save_json(report, report_path)
        
        print(f"📊 Báo cáo license plates: {report_path}")
        return report
    
    def export_face_vectors_csv(self):
        """Xuất face vectors dưới dạng CSV (meta info)"""
        import csv
        
        csv_path = self.face_vectors_dir / f"face_vectors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Vector_ID', 'Timestamp', 'Image_Path'])
            
            for name, data in self.faces_data.items():
                for vector_info in data.get('vectors', []):
                    writer.writerow([
                        name,
                        vector_info['id'],
                        vector_info['timestamp'],
                        vector_info['image_path']
                    ])
        
        print(f"📄 CSV file: {csv_path}")
        return csv_path
    
    def export_lp_csv(self):
        """Xuất license plates dưới dạng CSV"""
        import csv
        
        csv_path = self.lp_data_dir / f"license_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Plate_Text', 'Timestamp', 'Image_Path', 'Count'])
            
            for plate_text, data in self.lp_data.items():
                for img_info in data.get('images', []):
                    writer.writerow([
                        plate_text,
                        img_info['timestamp'],
                        img_info['image_path'],
                        data['count']
                    ])
        
        print(f"📄 CSV file: {csv_path}")
        return csv_path
    
    # ============ CHECK-IN METHODS ============

    def record_checkin(self, plate_text, face_name, metadata=None):
        """
        Ghi nhận CHECK-IN (time_in) cho một biển số
        """
        plate_text = plate_text.strip().upper()
        timestamp = datetime.now().isoformat()

        # Nếu chưa có plate thì tạo mới (KHÔNG coi là đã check-in)
        if plate_text not in self.lp_data:
            self.lp_data[plate_text] = {
                'plate_text': plate_text,
                'images': [],
                'checkins': [],
                'created_at': timestamp,
                'updated_at': timestamp,
                'count': 0
            }

        if 'checkins' not in self.lp_data[plate_text]:
            self.lp_data[plate_text]['checkins'] = []

        # ❗ CHỈ CẤM nếu có check-in CHƯA checkout
        for c in self.lp_data[plate_text]['checkins']:
            if c.get('status') == 'checked_in' and c.get('time_out') is None:
                print("⚠️ Plate already checked-in (open session)")
                return False

        # Tạo phiên check-in mới
        self.lp_data[plate_text]['checkins'].append({
            'face_name': face_name,
            'time_in': timestamp,
            'time_out': None,
            'duration_sec': None,
            'status': 'checked_in',
            'metadata': metadata or {}
        })

        self.lp_data[plate_text]['updated_at'] = timestamp
        self._save_json(self.lp_data, self.lp_db)

        print(f"🟢 CHECK-IN {plate_text} at {timestamp}")
        return True

    # ============ CHECK-OUT METHODS ============

    def record_checkout(self, plate_text):
        """
        Ghi nhận CHECK-OUT (time_out) cho biển số
        """
        plate_text = plate_text.strip().upper()
        timestamp = datetime.now().isoformat()

        if plate_text not in self.lp_data:
            print("❌ Plate not found for checkout")
            return None

        if 'checkins' not in self.lp_data[plate_text]:
            print("❌ No check-in history for plate")
            return None

        # Tìm check-in gần nhất chưa checkout
        for c in reversed(self.lp_data[plate_text]['checkins']):
            if c.get('status') == 'checked_in':
                c['time_out'] = timestamp
                c['status'] = 'checked_out'

                t_in = datetime.fromisoformat(c['time_in'])
                t_out = datetime.fromisoformat(timestamp)
                c['duration_sec'] = (t_out - t_in).total_seconds()

                self.lp_data[plate_text]['updated_at'] = timestamp
                self._save_json(self.lp_data, self.lp_db)

                print(f"🔴 CHECK-OUT {plate_text} at {timestamp}")
                return c

        print("⚠️ No active check-in found")
        return None

    
    # ============ UTILITY METHODS ============
    
    def list_saved_persons(self):
        """Liệt kê danh sách người đã lưu"""
        print("\n📋 Danh sách người đã lưu:")
        for i, (name, data) in enumerate(self.faces_data.items(), 1):
            print(f"   {i}. {name} - {data['count']} vectors")
    
    def list_saved_plates(self):
        """Liệt kê danh sách biển số đã lưu"""
        print("\n📋 Danh sách biển số đã lưu:")
        for i, (plate, data) in enumerate(self.lp_data.items(), 1):
            print(f"   {i}. {plate} - {data['count']} images")
    
    def get_summary(self):
        """Lấy tóm tắt dataset"""
        return {
            'faces': self.get_face_vector_stats(),
            'license_plates': self.get_license_plate_stats(),
            'directories': {
                'face_vectors': str(self.face_vectors_dir),
                'license_plates': str(self.lp_data_dir),
                'face_images': str(self.face_images_dir),
                'lp_images': str(self.lp_images_dir)
            }
        }


if __name__ == "__main__":
    # Test Dataset Manager
    manager = DatasetManager()
    
    print("\n" + "="*70)
    print("Dataset Manager - Test")
    print("="*70)
    
    # Show summary
    summary = manager.get_summary()
    print(f"\n📊 Tóm tắt dataset:")
    print(f"   👤 Người: {summary['faces']['total_persons']}")
    print(f"   🔢 Vectors: {summary['faces']['total_vectors']}")
    print(f"   🚗 Biển số: {summary['license_plates']['total_unique_plates']}")
    print(f"   📷 Ảnh biển số: {summary['license_plates']['total_images']}")
    
    # List saved data
    manager.list_saved_persons()
    manager.list_saved_plates()
    
    # Export reports
    manager.export_face_report()
    manager.export_lp_report()
    manager.export_face_vectors_csv()
    manager.export_lp_csv()
