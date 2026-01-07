import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Load raw data from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path)
    logger.info(f"✓ Loaded data: {df.shape} rows, {df.shape} columns")
    return df

def preprocess(df: pd.DataFrame) -> tuple:
    """
    Preprocess flight price data:
    - Handle missing values
    - Encode categorical variables
    - Scale numeric features
    - Create train/val/test splits
    
    Returns: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    
    df = df.copy()
    
    # Drop rows with missing values
    initial_shape = df.shape
    df = df.dropna()
    logger.info(f"Dropped {initial_shape - df.shape} rows with missing values")
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"After dedup: {df.shape} rows")
    
    # Separate target and features
    # Adjust 'Price' if your column name is different
    if 'Price' not in df.columns:
        # Find numeric column that looks like a price
        numeric_cols = df.select_dtypes(['int64', 'float64']).columns
        target_col = numeric_cols[-1]  # Assume last numeric is target
        logger.warning(f"'Price' column not found, using '{target_col}' as target")
    else:
        target_col = 'Price'
    
    y = df[target_col].copy()
    X = df.drop(target_col, axis=1)
    
    logger.info(f"Target variable: {target_col}, Range: [{y.min()}, {y.max()}]")
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes('object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        logger.info(f"Encoded categorical column: {col} ({len(le.classes_)} classes)")
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(['float64', 'int64']).columns.tolist()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    logger.info(f"Scaled {len(numeric_cols)} numeric columns")
    
    # Stratified train/val/test split (60/20/20)
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42,
            stratify=pd.cut(y, bins=5, duplicates='drop')
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=pd.cut(y_temp, bins=5, duplicates='drop')
        )
    except Exception as e:
        # If stratification fails, do random split
        logger.warning(f"Stratified split failed: {e}, using random split")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Save preprocessor for later use
    preprocessor = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'numeric_cols': numeric_cols,
        'feature_names': X.columns.tolist(),
        'target_col': target_col
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    logger.info("✓ Saved preprocessor to models/preprocessor.pkl")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    """Save train/val/test splits for model training."""
    os.makedirs('data/processed', exist_ok=True)
    
    splits = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    joblib.dump(splits, 'data/processed/splits.pkl')
    logger.info("✓ Saved data splits to data/processed/splits.pkl")

if __name__ == '__main__':
    # Run preprocessing pipeline
    logger.info("=" * 60)
    logger.info("STARTING DATA PREPROCESSING")
    logger.info("=" * 60)
    
    # Load data
    df = load_data('data/raw/flight_prices.csv')  # Adjust filename if different
    
    # Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess(df)
    
    # Save splits
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    
    logger.info("=" * 60)
    logger.info("✓ PREPROCESSING COMPLETE!")
    logger.info("=" * 60)
