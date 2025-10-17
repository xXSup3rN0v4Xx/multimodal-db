"""
API Dependencies
Shared dependencies for database connections and authentication.
"""
from functools import lru_cache
import sys
import os
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import MultimodalDB, QdrantVectorDB

# Global database instances
_db = None
_vector_db = None

def get_db():
    """Get database instance. Returns a fresh connection each time to pick up new data."""
    global _db
    if _db is None:
        # Use absolute path to top-level data folder
        # API is in: multimodal-db/multimodal-db/api/dependencies.py
        # Data is in: multimodal-db/data/multimodal_db/
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data" / "multimodal_db"
        _db = MultimodalDB(db_path=str(data_path))
        
        # Clean up any duplicate agents on startup
        print("Checking for duplicate agents...")
        _db.deduplicate_agents()
    return _db

@lru_cache()
def get_vector_db():
    """Get vector database instance with error handling."""
    global _vector_db
    if _vector_db is None:
        try:
            # Use a separate path for API to avoid conflicts with notebooks/tests
            _vector_db = QdrantVectorDB(persist_path="api_vectors")
            _vector_db.initialize_collections()
        except RuntimeError as e:
            if "already accessed" in str(e):
                print("⚠️  Warning: Qdrant storage locked by another process")
                print("   Vector search will be disabled. Close other instances to enable.")
                # Create a mock vector DB that's not available
                _vector_db = type('MockVectorDB', (), {
                    'available': False,
                    'collections': {},
                    'initialize_collections': lambda: False,
                    'get_stats': lambda: {'available': False, 'collections': {}},
                    'list_collections': lambda: []
                })()
            else:
                raise
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize vector database: {e}")
            _vector_db = type('MockVectorDB', (), {
                'available': False,
                'collections': {},
                'initialize_collections': lambda: False,
                'get_stats': lambda: {'available': False, 'collections': {}},
                'list_collections': lambda: []
            })()
    return _vector_db