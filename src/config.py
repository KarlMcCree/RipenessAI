from pathlib import Path

DATA_ROOT   = Path("data")
REAL_DIR    = DATA_ROOT / "real"        
AUGMENTED_DIR = DATA_ROOT / "augmented" 
DATA_DIR    = REAL_DIR                  

CLASS_NAMES = ["Green", "Semi-ripe", "Ripe", "Overripe"]
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
SEED        = 42
SPLIT       = 0.30

REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)
MODELS  = Path("models");  MODELS.mkdir(exist_ok=True)