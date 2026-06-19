import os, random, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)   # 1. Python hashing
    random.seed(seed)                          # 2. Python random
    np.random.seed(seed)                       # 3. NumPy
    # 4. framework + GPU (uncomment if using PyTorch):
    # import torch
    # torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)

def main(seed=42):
    set_seed(seed)
    df = pd.read_csv('data/breast_cancer.csv')
    X = df.drop(columns=['target']).values
    y = df['target'].values

    # SPLIT FIRST -- then fit the scaler on TRAIN only
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    clf = LogisticRegression(max_iter=200, random_state=seed).fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f'seed={seed}  accuracy={acc:.4f}')
    return clf, scaler, acc

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args().seed)
