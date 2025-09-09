import ast
import csv
import numpy as np
from diseases import diseases
from sklearn.metrics import *
from matplotlib import pyplot as plt

y_true = []
y_pred = []

MODEL = 'gpt-4o'
MODE = 'multichoice' #or multichoice

with open(f'{MODE}/{MODEL}.csv') as f: 
    reader = csv.DictReader(f)
    for row in reader: 
        y_true.append(ast.literal_eval(row['true']))
        y_pred.append(ast.literal_eval(row['response']))

# Define all possible labels
all_labels = list(diseases.keys())
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

# Convert to binary arrays
def binarize(labels):
    arr = np.zeros(len(all_labels))
    for label in labels:
        arr[label_to_idx[label.strip()]] = 1
    return arr


y_true = np.array([binarize(labels) for labels in y_true])
y_pred = np.array([binarize(labels) for labels in y_pred])

names = [diseases[label] for label in all_labels]
print(classification_report(y_true, y_pred, target_names=names, zero_division=0.0))

print("")
print("Hamming Loss: ", hamming_loss(y_true, y_pred))
print("Subset Accuracy: ", accuracy_score(y_true, y_pred))