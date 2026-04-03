import pandas as pd
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.getcwd())

from backend.vector_store import PatentVectorStore

def test_mapping_logic():
    print("Testing PatentVectorStore column remapping and IPC/CPC generation...")
    
    # We use the real CSV
    vs = PatentVectorStore(csv_path="data/patents_clean.csv")
    
    # Check if df is loaded
    if vs.df is None:
        print("FAILED: DataFrame is None")
        return
    
    # Check columns
    expected_cols = ['year', 'applicant', 'ipc_cpc', 'title', 'abstract']
    missing = [c for c in expected_cols if c not in vs.df.columns]
    
    if missing:
        print(f"FAILED: Missing columns: {missing}")
        print(f"Available columns: {vs.df.columns.tolist()}")
    else:
        print(f"SUCCESS: Found all expected columns.")
        
        # Check IPC/CPC generation
        print(f"Sample IPC/CPC: {vs.df['ipc_cpc'].head().tolist()}")
        if "UNKNOWN" in vs.df['ipc_cpc'].values:
            print("INFO: Found some UNKNOWN IPC/CPC values.")
            
        print("Test passed!")

if __name__ == "__main__":
    test_mapping_logic()
