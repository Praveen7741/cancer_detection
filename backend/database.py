from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# MySQL Connection Configuration
# NOTE: Replace 'root', 'your_password', and 'cancer_detection_db' with your actual MySQL details!
DATABASE_URL = "mysql+pymysql://root:praveen@localhost:3306/cancer_detection_db"

# Create the MySQL engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PatientRecord(Base):
    __tablename__ = "patient_triage_records" # Changed to create a fresh table with correct memory limits

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # We store the image filename or a UUID to track batches
    filename = Column(String(255), index=True)
    
    # Model Predictions
    subtype = Column(String(50))
    confidence = Column(Float)
    recurrence_risk = Column(Float)
    risk_group = Column(String(50))
    
    # Store the base64 string directly (length set to 4.2 Billion to force MySQL LONGTEXT data type instead of standard 64kb TEXT)
    gradcam_base64 = Column(Text(4294967295))

# Initialize the database tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
