import time
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# Đo thời gian load data
start_time = time.time()
data = pd.read_csv('/home/ubuntu/ml-benchmark/creditcard.csv')
load_time = time.time() - start_time

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Huấn luyện model
clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
start_time = time.time()
clf.fit(X_train, y_train)
train_time = time.time() - start_time

# Đánh giá trên tập test
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

# Đo inference latency (1 dòng)
single_row = X_test.iloc[[0]]
start_time = time.time()
clf.predict(single_row)
latency = (time.time() - start_time) * 1000 # milliseconds

# Đo inference throughput (1000 dòng)
thousand_rows = X_test.iloc[:1000]
start_time = time.time()
clf.predict(thousand_rows)
throughput_time = time.time() - start_time
throughput = 1000 / throughput_time if throughput_time > 0 else 0

results = {
    "Load Data Time (s)": round(load_time, 4),
    "Training Time (s)": round(train_time, 4),
    "Best Iteration": clf.best_iteration_ if clf.best_iteration_ else "N/A",
    "AUC-ROC": round(auc, 4),
    "Accuracy": round(acc, 4),
    "F1-Score": round(f1, 4),
    "Precision": round(prec, 4),
    "Recall": round(rec, 4),
    "Inference Latency 1 row (ms)": round(latency, 4),
    "Inference Throughput 1000 rows (rows/sec)": round(throughput, 2)
}

print(json.dumps(results, indent=4))

with open('benchmark_result.json', 'w') as f:
    json.dump(results, f, indent=4)
