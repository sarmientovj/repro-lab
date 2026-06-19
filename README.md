# Reproducible ML Pipeline — Session 5

## What this is
A logistic-regression classifier on the Iris dataset, built to be
fully reproducible: versioned code, data, experiments and environment.

## Data
data/breast_cancer.csv is tracked with DVC. Pointer file: data/breast_cancer.csv.dvc

## Reproduce the result
1. pip install -r requirements.txt
2. dvc pull            # retrieve the exact dataset version
3. python src/train.py --seed 42

## Expected output
seed=42  accuracy=0.9737

## Environment
Python 3.11; exact packages in requirements.txt; see Dockerfile.
