import os, random, argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

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
    df = pd.read_csv('data/iris.csv')
    X = df.drop(columns=['target']).values
    y = df['target'].values

    # SPLIT FIRST -- then fit the scaler on TRAIN only
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    clf = LogisticRegression(max_iter=200, random_state=seed).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f'seed={seed}  accuracy={acc:.4f}')
    return clf, scaler, acc

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args().seed)
