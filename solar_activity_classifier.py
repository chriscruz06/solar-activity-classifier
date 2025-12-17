"""
Solar Activity Classifier
A rudimentary cyberinfrastructure-enabled machine learning tool for classifying solar activity levels

Christopher Cruz & Ameer Hassan
NJIT
December 2025

This tool uses Random Forest classification to predict solar activity levels
based on sunspot data and solar indices.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class SolarActivityClassifier:
    """
    A machine learning classifier for solar activity prediction.
    
    This tool classifies solar activity into three categories:
    - "Low": Light sun conditions
    - "Medium": Moderate activity
    - "High": Active sun with potential for flares
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the Solar Activity Classifier
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'sunspot_number',
            'sunspot_area',
            'new_regions',
            'solar_flux_10.7cm',
            'prev_day_activity'
        ]
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic solar activity data for demonstration
        
        In a real deployment, this would be replaced with actual
        solar observation data from NOAA, SDO/HMI, or other sources.
        """
        np.random.seed(42)
        
        # Generate features with realistic correlations
        sunspot_number = np.random.gamma(shape=2, scale=50, size=n_samples)
        sunspot_area = sunspot_number * np.random.uniform(0.8, 1.2, n_samples)
        new_regions = np.random.poisson(lam=sunspot_number/50, size=n_samples)
        solar_flux = 65 + sunspot_number * 0.5 + np.random.normal(0, 10, n_samples)
        prev_day = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
        
        X = np.column_stack([
            sunspot_number,
            sunspot_area,
            new_regions,
            solar_flux,
            prev_day
        ])
        
        # Generate labels based on activity level
        y = np.zeros(n_samples, dtype=int)
        y[sunspot_number > 50] = 1  # Medium
        y[sunspot_number > 100] = 2  # High
        
        return X, y
    
    def train(self, X, y):
        """
        Train the classifier
        """
        print("Training Solar Activity Classifier...")
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_names)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_scaled)
        train_acc = accuracy_score(y, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
        print("Training complete!")
        
    def predict(self, X):
        """
        Make predictions on new data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
            
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        """
        predictions, probabilities = self.predict(X)
        
        results = {
            'accuracy': accuracy_score(y, predictions),
            'confusion_matrix': confusion_matrix(y, predictions),
            'classification_report': classification_report(
                y, predictions,
                target_names=['Low', 'Medium', 'High'],
                output_dict=True
            )
        }
        
        return results
    
    def plot_confusion_matrix(self, cm, title='Confusion Matrix'):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_feature_importance(self):
        """
        Plot feature importance from the trained model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
            
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance in Solar Activity Classification')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), 
                   [self.feature_names[i] for i in indices],
                   rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.tight_layout()
        return plt.gcf()
    
    def save_model(self, filepath='models/solar_activity_classifier.pkl'):
        """
        Save trained model to disk
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")
            
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/solar_activity_classifier.pkl'):
        """
        Load trained model from disk
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def main():
    """
    Main execution function demonstrating the tool's capabilities
    """
    print("=" * 70)
    print("Solar Activity Classifier")
    print("Cyberinfrastructure-Enabled Machine Learning Tool")
    print("=" * 70)
    print()
    
    # Initialize classifier
    classifier = SolarActivityClassifier(n_estimators=100)
    
    # Generate data (in production, load real solar data)
    print("Generating synthetic solar activity data...")
    X, y = classifier.generate_synthetic_data(n_samples=1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print()
    
    # Train model
    classifier.train(X_train, y_train)
    print()
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    results = classifier.evaluate(X_test, y_test)
    
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print("-" * 50)
    
    report = results['classification_report']
    for class_name in ['Low', 'Medium', 'High']:
        metrics = report[class_name]
        print(f"{class_name:10s} - Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}, "
              f"F1-Score: {metrics['f1-score']:.3f}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return classifier, results


if __name__ == "__main__":
    classifier, results = main()