from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path
import config

MAX_FILE_SIZE = 1_000_000  # 1 MB

def load_docs(data_path=None):
    if data_path is None:
        data_path = config.DATA_PATH
    
    docs = []
    data_path_obj = Path(data_path)
    
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    files = list(data_path_obj.rglob("*"))
    valid_files = [f for f in files if f.is_file() and f.suffix.lower() in [".pdf", ".txt", ".md"]]
    total_files = len(valid_files)

    print(f"Found {total_files} valid files. Skipping >1MB...")
    
    processed = 0
    for file in valid_files:
        # Skip files larger than 1 MB
        if file.stat().st_size > MAX_FILE_SIZE:
            print(f"Skipping {file.name} (size > 1MB)")
            continue
        
        try:
            if file.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file))
            else:
                loader = TextLoader(str(file), encoding="utf-8")
            
            docs.extend(loader.load())
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_files} ..")
        
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    print(f"Loaded {len(docs)} documents from {processed} files")
    return docs
