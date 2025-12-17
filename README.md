
# Solar Activity Classifier

A rudimentary cyberinfrastructure-enabled machine learning tool for classifying solar activity levels.

## Authors

**Christopher Cruz & Ameer Hassan**  
New Jersey Institute of Technology  
December 2025

## Abstract

Solar activity classification is crucial for space weather forecasting and understanding solar variability. This tool employs a Random Forest machine learning classifier to categorize solar conditions into three activity levels (Low, Medium, High) based on observable solar parameters including sunspot numbers, solar flux measurements, and active region characteristics.

The tool is designed following the cyberinfrastructure principles established by the Dr. Jason T. L. Wang and his team, enabling easy deployment, reproducibility, and accessibility through Jupyter notebook interfaces.

## Features

- **Multi-class Classification**: Categorizes solar activity into Low/Medium/High levels
- **Feature Importance Analysis**: Identifies key predictors of solar activity
- **Visualization Tools**: Comprehensive plotting of results and model performance
- **Model Persistence**: Save and load trained models for operational use
- **Jupyter Interface**: Interactive notebook for easy experimentation
- **Binder-Ready**: Can be deployed on cloud infrastructure

## Installation

### Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

### Quick Start

1. Clone this repository:
```bash
git clone https://github.com/chriscruz06/solar-activity-classifier.git
cd solar-activity-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook solar_activity_classifier.ipynb
```

### Using Binder

Click the badge below to run this notebook in your browser without installation:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/your-username/solar-activity-classifier/HEAD?labpath=solar_activity_classifier.ipynb)

## Usage

### Basic Example

```python
from solar_activity_classifier import SolarActivityClassifier

# Initialize classifier
classifier = SolarActivityClassifier(n_estimators=100)

# Generate or load data
X, y = classifier.generate_synthetic_data(n_samples=1000)

# Train model
classifier.train(X, y)

# Make predictions
predictions, probabilities = classifier.predict(X_new)
```

### Running from Command Line

```bash
python solar_activity_classifier.py
```

## Methodology

### Features Used

1. **Sunspot Number**: Total count of sunspots observed
2. **Sunspot Area**: Total area covered by sunspots (millionths of solar hemisphere)
3. **New Active Regions**: Number of newly emerged active regions
4. **Solar Flux (10.7cm)**: Radio flux measurement at 10.7cm wavelength
5. **Previous Day Activity**: Activity level from the previous day

### Classification Scheme

- **Low Activity** (Class 0): Sunspot number < 50
- **Medium Activity** (Class 1): 50 ≤ Sunspot number < 100
- **High Activity** (Class 2): Sunspot number ≥ 100

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Max Depth**: 10
- **Feature Scaling**: StandardScaler normalization

## Performance Metrics

On test data (20% split):
- **Overall Accuracy**: ~85-90%
- **Precision/Recall**: Balanced across all classes
- **Feature Importance**: Sunspot-related features most predictive

## File Structure

```
solar-activity-classifier/
├── solar_activity_classifier.py      # Main Python module
├── solar_activity_classifier.ipynb   # Jupyter notebook interface
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
```

## Comparison with CCSC Tools

This tool follows the design patterns established by the NJIT CCSC tools (Provided to us by Dr. Jason T. L. Wang, wangj@njit.edu):

| Aspect | CCSC Tools | This Tool |
|--------|-----------|-----------|
| Interface | Jupyter notebooks | Jupyter notebook |
| Binder Support | Yes | Yes |
| ML Framework | TensorFlow/Scikit-learn | Scikit-learn |
| Visualization | Matplotlib/Seaborn | Matplotlib/Seaborn |
| Model Saving | Pickle/HDF5 | Joblib |

## Future Enhancements

1. **Real Data Integration**: Connect to NOAA/NASA data APIs
2. **Temporal Forecasting**: Add LSTM/RNN for time-series prediction
3. **Web Dashboard**: Develop Flask/Dash interface for operational use
4. **Ensemble Methods**: Combine with other ML algorithms
5. **Uncertainty Quantification**: Add Bayesian inference capabilities

## Related CCSC Tools

This tool complements existing CCSC solar ML tools:

- **FlareML**: Multi-class flare prediction
- **LSTM-Flare**: Time-series flare forecasting  
- **SolarUnet**: Magnetic flux tracking
- **CMETNet**: CME arrival time prediction

## References

1. Community Coordinated Software Center (CCSC). https://nature.njit.edu/solardb/ccsc
2. NOAA Space Weather Prediction Center. https://www.swpc.noaa.gov/
3. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12:2825-2830.


## Acknowledgments

- NJIT Community Coordinated Software Center
- Professor Dr. Jason T. L. Wang [wangj@njit.edu]

## Contact

For questions or anything else:
- Emails: [cac245@njit.edu] [amh23@njit.edu]
- GitHub: [@chriscruz06]

---

**Developed as part of NJIT's research coursework**