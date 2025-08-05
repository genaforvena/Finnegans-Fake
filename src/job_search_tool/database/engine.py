from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from job_search_tool.config import load_config
from job_search_tool.database.models import Base

def get_engine():
    """
    Creates a SQLAlchemy engine based on the configuration.
    """
    config = load_config()
    db_name = config["database"]["db_name"]
    db_url = f"sqlite:///{db_name}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    return engine

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Initializes the database and creates tables.
    """
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    print("Initializing the database...")
    init_db()
    print("Database initialized successfully.")
