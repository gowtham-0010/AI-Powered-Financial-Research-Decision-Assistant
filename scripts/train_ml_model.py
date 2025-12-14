"""
Script to train ML risk classifier model.
Loads financial data, trains Random Forest classifier, and saves artifacts.
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_financial_data() -> pd.DataFrame:
    """
    Load cleaned financial data from CSV.
    
    Returns:
        pd.DataFrame: Financial data with company metrics
    """
    data_path = Path("data/processed/financial_data_clean.csv")
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Please run scripts/prepare_data.py first")
    
    logger.info(f"Loading financial data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    return df


def create_risk_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Create risk level labels from financial data.
    
    Risk classification logic:
    - LOW: Strong fundamentals (high ROE, low debt, good liquidity)
    - MEDIUM: Average metrics
    - HIGH: Weak fundamentals (low ROE, high debt, low liquidity)
    
    Scoring system:
    - P/E Ratio: Lower is better (risk decreases at P/E < 15)
    - Debt/Equity: Lower is better (risk increases > 1.5)
    - Current Ratio: Higher is better (risk high < 1.0)
    - ROE: Higher is better (risk low > 0.15)
    - Beta: Lower is better (risk high > 1.5)
    - Revenue Growth: Higher is better (risk high < -0.05)
    - Sector Risk: Lower is better (risk high > 0.7)
    
    Args:
        df: DataFrame with financial metrics
        
    Returns:
        np.ndarray: Risk labels (0=LOW, 1=MEDIUM, 2=HIGH)
    """
    logger.info("Creating risk labels from financial metrics")
    
    # Initialize risk scores (0=low risk, higher=high risk)
    risk_scores = np.zeros(len(df))
    
    # P/E Ratio score (5-50 typical range)
    pe_ratio = df['pe_ratio'].fillna(df['pe_ratio'].mean())
    risk_scores += np.where(pe_ratio > 30, 2, np.where(pe_ratio > 20, 1, 0))
    
    # Debt/Equity score (lower is better)
    debt_equity = df['debt_equity'].fillna(df['debt_equity'].mean())
    risk_scores += np.where(debt_equity > 2.0, 2, np.where(debt_equity > 1.0, 1, 0))
    
    # Current Ratio score (higher is better for liquidity)
    current_ratio = df['current_ratio'].fillna(df['current_ratio'].mean())
    risk_scores += np.where(current_ratio < 1.0, 2, np.where(current_ratio < 1.5, 1, 0))
    
    # ROE score (higher is better)
    roe = df['roe'].fillna(df['roe'].mean())
    risk_scores += np.where(roe < 0.05, 2, np.where(roe < 0.15, 1, 0))
    
    # Beta score (lower is better - less volatile)
    beta = df['beta'].fillna(df['beta'].mean())
    risk_scores += np.where(beta > 1.5, 2, np.where(beta > 1.0, 1, 0))
    
    # Revenue Growth score (higher is better)
    revenue_growth = df['revenue_growth'].fillna(df['revenue_growth'].mean())
    risk_scores += np.where(revenue_growth < -0.05, 2, np.where(revenue_growth < 0.05, 1, 0))
    
    # Sector Risk score (lower is better)
    sector_risk = df['sector_risk'].fillna(df['sector_risk'].mean())
    risk_scores += np.where(sector_risk > 0.7, 2, np.where(sector_risk > 0.4, 1, 0))
    
    # Convert scores to labels: 0=LOW, 1=MEDIUM, 2=HIGH
    # Normalize risk scores (0-14 possible range)
    risk_scores_normalized = risk_scores / 14.0
    
    # Assign labels based on thresholds
    labels = np.zeros(len(df), dtype=int)
    labels[risk_scores_normalized > 0.5] = 2  # HIGH
    labels[(risk_scores_normalized > 0.3) & (risk_scores_normalized <= 0.5)] = 1  # MEDIUM
    
    logger.info(f"Risk label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = ['LOW', 'MEDIUM', 'HIGH'][label]
        logger.info(f"  {label_name}: {count} ({count/len(labels)*100:.1f}%)")
    
    return labels


def prepare_features_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and create labels for training.
    
    Args:
        df: DataFrame with financial metrics
        
    Returns:
        Tuple of (X, y) where X is features array and y is labels array
    """
    # Feature names matching ml_models/risk_classifier.py
    feature_names = [
        'pe_ratio',
        'debt_equity',
        'current_ratio',
        'roe',
        'beta',
        'revenue_growth',
        'sector_risk'
    ]
    
    logger.info(f"Extracting {len(feature_names)} features: {feature_names}")
    
    # Extract features
    X = df[feature_names].fillna(df[feature_names].mean()).values
    
    # Create labels
    y = create_risk_labels(df)
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    return X, y


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train Random Forest classifier and evaluate.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info("Training Random Forest classifier...")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle imbalanced classes
    )
    
    # Train model
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred,
        target_names=['LOW', 'MEDIUM', 'HIGH']
    )
    
    # Log results
    logger.info(f"Model Evaluation Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1-Score (weighted): {f1:.4f}")
    logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
    logger.info(f"\nClassification Report:\n{class_report}")
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }
    
    return model, metrics


def save_model_artifacts(
    model: RandomForestClassifier,
    scaler: StandardScaler
) -> None:
    """
    Save trained model and scaler to disk.
    
    Args:
        model: Trained RandomForestClassifier
        scaler: Fitted StandardScaler
    """
    model_dir = Path("ml_models/trained_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / "risk_classifier_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = model_dir / "feature_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save feature names
    feature_names_path = model_dir / "feature_names.pkl"
    feature_names = [
        'pe_ratio',
        'debt_equity',
        'current_ratio',
        'roe',
        'beta',
        'revenue_growth',
        'sector_risk'
    ]
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info(f"Feature names saved to {feature_names_path}")


def main():
    """Main training pipeline."""
    logger.info("Starting ML model training pipeline...")
    
    try:
        # 1. Load data
        df = load_financial_data()
        
        # 2. Prepare features and labels
        X, y = prepare_features_and_labels(df)
        
        # 3. Split data with intelligent stratification handling
        logger.info("Splitting data: 80% train, 20% test")
        
        # Check if stratified split is viable
        n_samples = len(X)
        n_classes = len(np.unique(y))
        test_size = 0.2
        min_test_samples = int(np.ceil(n_samples * test_size))
        
        # Determine if stratification is possible
        # Stratification requires at least n_classes samples in test set
        use_stratify = min_test_samples >= n_classes
        
        if not use_stratify:
            logger.warning(
                f"Dataset too small for stratified split ({n_samples} samples, "
                f"{n_classes} classes, would have {min_test_samples} test samples). "
                f"Using non-stratified split instead."
            )
            # Non-stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            # Stratified split
            logger.info(f"Using stratified split ({n_samples} samples sufficient for {n_classes} classes)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # 4. Scale features
        logger.info("Scaling features with StandardScaler")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 5. Train model
        model, metrics = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # 6. Save artifacts
        save_model_artifacts(model, scaler)
        
        logger.info("✅ Training pipeline complete!")
        logger.info(f"Model ready for inference")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()