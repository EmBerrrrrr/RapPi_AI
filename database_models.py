"""
PostgreSQL Database Models for Parking Management System
Matches with .NET Backend Schema
"""

from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, ForeignKey, Text, Date, Time, DECIMAL, JSON
from sqlalchemy.dialects.postgresql import UUID, BYTEA, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid

Base = declarative_base()


class ParkingLot(Base):
    """Parking lot model"""
    __tablename__ = 'parking_lots'
    
    lot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lot_code = Column(String(50), unique=True, nullable=False)
    lot_name = Column(String(200), nullable=False)
    full_address = Column(Text, nullable=False)
    total_capacity = Column(Integer, nullable=False)
    current_occupancy = Column(Integer, default=0)
    hourly_rate = Column(DECIMAL(10, 2), nullable=False)
    daily_rate = Column(DECIMAL(10, 2))
    monthly_rate = Column(DECIMAL(10, 2))
    free_minutes = Column(Integer)
    opening_time = Column(Time, default='06:00:00')
    closing_time = Column(Time, default='22:00:00')
    is_24h = Column(Boolean, default=False)
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    parking_sessions = relationship("ParkingSession", back_populates="parking_lot")
    iot_devices = relationship("IoTDevice", back_populates="parking_lot")


class Staff(Base):
    """Staff model"""
    __tablename__ = 'staffs'
    
    staff_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lot_id = Column(UUID(as_uuid=True), ForeignKey('parking_lots.lot_id'), nullable=False)
    staff_code = Column(String(50), unique=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    face_encoding = Column(BYTEA, nullable=False)
    face_image_url = Column(Text)
    employment_status = Column(String(20), default='active')
    phone_contact = Column(String(20))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class User(Base):
    """User model"""
    __tablename__ = 'users'
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phone_number = Column(String(15), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date)
    is_active = Column(Boolean, default=True)
    avatar_url = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    vehicles = relationship("Vehicle", back_populates="user")
    parking_sessions = relationship("ParkingSession", back_populates="user")
    wallet = relationship("Wallet", back_populates="user", uselist=False)
    loyalty_points = relationship("LoyaltyPoints", back_populates="user", uselist=False)
    monthly_passes = relationship("MonthlyPass", back_populates="user")


class Vehicle(Base):
    """Vehicle model"""
    __tablename__ = 'vehicles'
    
    vehicle_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False, index=True)
    license_plate = Column(String(20), unique=True, nullable=False, index=True)
    plate_image_url = Column(Text)
    vehicle_type = Column(String(50))
    color = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="vehicles")
    parking_sessions = relationship("ParkingSession", back_populates="vehicle")
    monthly_passes = relationship("MonthlyPass", back_populates="vehicle")


class IoTDevice(Base):
    """IoT Device model"""
    __tablename__ = 'iot_devices'
    
    device_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lot_id = Column(UUID(as_uuid=True), ForeignKey('parking_lots.lot_id'), nullable=False, index=True)
    device_code = Column(String(100), unique=True, nullable=False)
    device_name = Column(String(200), nullable=False)
    device_type = Column(String(50), nullable=False, index=True)
    installation_location = Column(String(200))
    gate_number = Column(Integer)
    gate_direction = Column(String(10))
    model = Column(String(100))
    ip_address = Column(String(45))
    mac_address = Column(String(17))
    connection_status = Column(String(20), default='disconnected')
    status = Column(String(20), default='active', index=True)
    firmware_version = Column(String(50))
    last_heartbeat = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    parking_lot = relationship("ParkingLot", back_populates="iot_devices")
    device_events = relationship("DeviceEvent", back_populates="device")


class ParkingSession(Base):
    """Parking session model"""
    __tablename__ = 'parking_sessions'
    
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lot_id = Column(UUID(as_uuid=True), ForeignKey('parking_lots.lot_id'), nullable=False, index=True)
    vehicle_id = Column(UUID(as_uuid=True), ForeignKey('vehicles.vehicle_id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False, index=True)
    
    check_in_time = Column(DateTime, nullable=False, index=True)
    check_in_face_image = Column(Text)
    check_in_plate_image = Column(Text)
    check_in_device = Column(UUID(as_uuid=True), ForeignKey('iot_devices.device_id'))
    
    check_out_time = Column(DateTime)
    check_out_face_image = Column(Text)
    check_out_plate_image = Column(Text)
    check_out_device = Column(UUID(as_uuid=True), ForeignKey('iot_devices.device_id'))
    
    face_match_score = Column(DECIMAL(5, 2))
    plate_match_score = Column(DECIMAL(5, 2))
    session_status = Column(String(20), default='parked', index=True)
    total_hours = Column(DECIMAL(10, 2))
    total_amount = Column(DECIMAL(12, 2))
    points_earned = Column(Integer, default=0)
    points_used = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    parking_lot = relationship("ParkingLot", back_populates="parking_sessions")
    vehicle = relationship("Vehicle", back_populates="parking_sessions")
    user = relationship("User", back_populates="parking_sessions")
    transactions = relationship("Transaction", back_populates="parking_session")
    recognition_logs = relationship("RecognitionLog", back_populates="session")


class Transaction(Base):
    """Transaction model"""
    __tablename__ = 'transactions'
    
    transaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('parking_sessions.session_id'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False, index=True)
    lot_id = Column(UUID(as_uuid=True), ForeignKey('parking_lots.lot_id'), nullable=False, index=True)
    
    transaction_type = Column(String(30), nullable=False)
    amount = Column(DECIMAL(12, 2), nullable=False)
    description = Column(Text)
    payment_method = Column(String(30), nullable=False)
    
    points_used = Column(Integer, default=0)
    points_earned = Column(Integer, default=0)
    
    momo_transaction_id = Column(String(100))
    vnpay_transaction_id = Column(String(100))
    payment_status = Column(String(20), default='pending')
    
    created_at = Column(DateTime, default=datetime.now, index=True)
    completed_at = Column(DateTime)
    
    # Relationships
    parking_session = relationship("ParkingSession", back_populates="transactions")


class Wallet(Base):
    """Wallet model"""
    __tablename__ = 'wallets'
    
    wallet_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), unique=True, nullable=False, index=True)
    balance = Column(DECIMAL(12, 2), default=0.00)
    currency = Column(String(10), default='VND')
    last_updated = Column(DateTime, default=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="wallet")


class LoyaltyPoints(Base):
    """Loyalty points model"""
    __tablename__ = 'loyalty_points'
    
    point_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), unique=True, nullable=False, index=True)
    total_earned = Column(Integer, default=0)
    total_used = Column(Integer, default=0)
    current_balance = Column(Integer, default=0)
    member_tier = Column(String(20), default='bronze')
    expiring_points = Column(Integer, default=0)
    next_expiry_date = Column(Date)
    last_updated = Column(DateTime, default=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="loyalty_points")


class MonthlyPass(Base):
    """Monthly pass model"""
    __tablename__ = 'monthly_passes'
    
    pass_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False, index=True)
    vehicle_id = Column(UUID(as_uuid=True), ForeignKey('vehicles.vehicle_id'), nullable=False)
    lot_id = Column(UUID(as_uuid=True), ForeignKey('parking_lots.lot_id'), nullable=False, index=True)
    
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    original_price = Column(DECIMAL(12, 2))
    paid_amount = Column(DECIMAL(12, 2))
    points_used = Column(Integer, default=0)
    payment_method = Column(String(30), default='cash')
    status = Column(String(20), default='active', index=True)
    qr_code = Column(Text)
    qr_code_expiry = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="monthly_passes")
    vehicle = relationship("Vehicle", back_populates="monthly_passes")


class RecognitionLog(Base):
    """Recognition log model"""
    __tablename__ = 'recognition_logs'
    
    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('parking_sessions.session_id'), index=True)
    recognition_type = Column(String(20))
    input_image_url = Column(Text)
    result_data = Column(JSONB)
    confidence_score = Column(DECIMAL(5, 2))
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.now, index=True)
    
    # Relationships
    session = relationship("ParkingSession", back_populates="recognition_logs")


class DeviceEvent(Base):
    """Device event model"""
    __tablename__ = 'device_events'
    
    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(UUID(as_uuid=True), ForeignKey('iot_devices.device_id'), nullable=False, index=True)
    lot_id = Column(UUID(as_uuid=True), ForeignKey('parking_lots.lot_id'), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('parking_sessions.session_id'))
    
    event_type = Column(String(50), nullable=False, index=True)
    event_data = Column(JSONB)
    severity = Column(String(20), default='info')
    created_at = Column(DateTime, default=datetime.now, index=True)
    
    # Relationships
    device = relationship("IoTDevice", back_populates="device_events")


# Database connection functions
def create_db_engine(
    host=None,
    port=5432,
    database='postgres',
    username='postgres',
    password=None
):
    """
    Create PostgreSQL database engine (Supabase compatible)
    
    Args:
        host: Database host (from .env)
        port: Database port
        database: Database name
        username: Database username
        password: Database password (from .env)
    
    Returns:
        SQLAlchemy engine and session maker
    """
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get connection details from .env
    host = host or os.getenv('DB_HOST')
    password = password or os.getenv('DB_PASSWORD')
    
    if not host or not password:
        raise ValueError("Database credentials not found in .env file")
    
    # Build connection URL with psycopg3 (supports IPv6)
    database_url = f"postgresql+psycopg://{username}:{password}@{host}:{port}/{database}"
    
    # Create engine with SSL for Supabase
    engine = create_engine(
        database_url, 
        echo=False,
        pool_pre_ping=True,
        connect_args={
            "sslmode": "require",  # Supabase requires SSL
            "connect_timeout": 10
        }
    )
    
    # Test connection
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        print(f"✅ Supabase PostgreSQL connected: {host}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        raise
    
    # Create tables if not exist
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    return engine, Session


def get_db_session(Session):
    """Get database session with context manager"""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
