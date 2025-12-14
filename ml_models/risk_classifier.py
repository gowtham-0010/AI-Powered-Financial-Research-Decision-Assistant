"""
ML Model: Risk Classifier for investment risk prediction.
Uses Random Forest to classify investment risk as Low/Medium/High.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from config.settings import get_settings

logger = logging.getLogger(__name__)


class RiskClassifier:
    """
    Machine learning risk classifier using Random Forest.
    
    Features:
    - PE Ratio
    - Debt-to-Equity Ratio
    - Current Ratio (Liquidity)
    - ROE (Return on Equity)
    - Beta (Volatility)
    - Revenue Growth YoY
    - Sector Risk Factor
    
    Classes:
    - Low Risk
    - Medium Risk
    - High Risk
    """
    
    FEATURE_NAMES = [
        'pe_ratio',
        'debt_equity',
        'current_ratio',
        'roe',
        'beta',
        'revenue_growth',
        'sector_risk'
    ]
    
    RISK_LEVELS = ['LOW', 'MEDIUM', 'HIGH']
    
    def __init__(self):
        """Initialize classifier and load model."""
        logger.info("Initializing RiskClassifier...")
        self.settings = get_settings()
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and scaler from disk."""
        model_path = Path(self.settings.ml_model_path) / self.settings.risk_model_name
        scaler_path = Path(self.settings.ml_model_path) / self.settings.feature_scaler_name
        
        try:
            if model_path.exists() and scaler_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Model and scaler loaded successfully")
            else:
                logger.warning("Model files not found. Using untrained model.")
                self._create_default_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}. Using default.")
            self._create_default_model()
    
    def _create_default_model(self):
        """Create default untrained model."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def predict(
        self,
        company_metrics: Dict[str, float]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict investment risk for company.
        
        Args:
            company_metrics: Dictionary with keys matching FEATURE_NAMES
                            {
                                'pe_ratio': 28.4,
                                'debt_equity': 0.89,
                                'current_ratio': 1.5,
                                'roe': 0.89,
                                'beta': 1.15,
                                'revenue_growth': 0.153,
                                'sector_risk': 0.3
                            }
        
        Returns:
            Tuple[str, float, Dict]: (risk_level, confidence, importances)
                - risk_level: 'LOW', 'MEDIUM', 'HIGH'
                - confidence: 0.0-1.0
                - importances: {feature: importance}
        """
        try:
            # Extract features in correct order
            features = np.array([
                [company_metrics.get(name, 0.0) for name in self.FEATURE_NAMES]
            ])
            
            # Normalize features
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Predict
            risk_index = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            risk_level = self.RISK_LEVELS[risk_index]
            confidence = float(probabilities[risk_index])
            
            # Feature importances
            importances = {
                name: float(imp)
                for name, imp in zip(self.FEATURE_NAMES, self.model.feature_importances_)
            }
            
            logger.info(
                f"Risk prediction: {risk_level} (confidence: {confidence:.2%})"
            )
            
            return risk_level, confidence, importances
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 'UNKNOWN', 0.0, {}
    
    def get_feature_importance_text(
        self,
        importances: Dict[str, float]
    ) -> str:
        """Format feature importances for display."""
        sorted_features = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        text = "Risk Factors (by importance):\n"
        for feature, importance in sorted_features[:5]:
            text += f"  • {feature.replace('_', ' ').title()}: {importance:.1%}\n"
        
        return text


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[RandomForestClassifier, StandardScaler, Dict[str, Any]]:
    """
    Train Risk Classifier model.
    
    Args:
        X_train: Training features (n_samples, 7)
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple: (model, scaler, metrics)
    """
    logger.info("Training RiskClassifier model...")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = model.score(X_test_scaled, y_test)
    
    metrics = {
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    logger.info(f"Training complete. F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    return model, scaler, metrics


# Global instance
_risk_classifier = None

def get_risk_classifier() -> RiskClassifier:
    """Get or create risk classifier (singleton)."""
    global _risk_classifier
    if _risk_classifier is None:
        _risk_classifier = RiskClassifier()
    return _risk_classifier


if __name__ == "__main__":
    # Test prediction
    classifier = get_risk_classifier()
    
    test_metrics = {
        'pe_ratio': 28.4,
        'debt_equity': 0.89,
        'current_ratio': 1.5,
        'roe': 0.89,
        'beta': 1.15,
        'revenue_growth': 0.153,
        'sector_risk': 0.3
    }
    
    risk_level, confidence, importances = classifier.predict(test_metrics)
    print(f"Risk Level: {risk_level}")
    print(f"Confidence: {confidence:.2%}")
    print(classifier.get_feature_importance_text(importances))
