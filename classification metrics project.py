# Importing libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, roc_auc_score, precision_recall_curve, roc_curve, auc
from datetime import datetime

# Open and read the file
input_ = input("Press any key to start")

df = pd.read_csv("KM-03-2.csv")
df.head()

# Check if Model_1_0+Model_1_1 and Model_2_0+Model_2_1 equal 1
input_ = input("Press any key to start")

df["prob_sum1model"] = 1 - df["Model_1_1"]
mask = np.isclose(df['Model_1_0'].astype(float), df["prob_sum1model"].astype(float))
not_matching_count = (~mask).sum()
print(not_matching_count)

df["prob_sum1mode2"] = 1 - df["Model_2_1"]
mask = np.isclose(df["Model_2_0"].astype(float), df["prob_sum1mode2"].astype(float))
not_matching_count = (~mask).sum()
print(not_matching_count)

# Data balance 1/0 quantity
input_ = input("Press any key to start")

value_1 = len(df[df["GT"] == 1])
value_0 = len(df[df["GT"] == 0])
print(value_1)
print(value_0)

balance_ratio = value_1/value_0
print(balance_ratio)

input("Press any key to start")
def metrics_comp(df_new_func):
  df = df_new_func
  cols = ["Model_1_1", "Model_2_1"]
  metrics = {"Accuracy": accuracy_score,
             "Precision": precision_score,
             "Recall": recall_score,
             "F-score": f1_score,
             "MCC": matthews_corrcoef,
             "Balanced Accuracy": balanced_accuracy_score,
             "Youden's J statistic": lambda y_true, y_pred: (sensitivity + specificity - 1),
             "AUC for Precision-Recall Curve": average_precision_score,
             "AUC for Receiver Operating Characteristic Curve": roc_auc_score}

  fig, ax = plt.subplots(figsize=(10, 6))
  for col in cols:
    print(df[col].name)
    thresholds = np.linspace(0.1, 0.99, 10)
    results = {}
    for i in thresholds:
      print(round(i, 2))
      df = df_new_func
      y_true = df["GT"]
      y_pred = df[col]
      y_pred = y_pred.apply(lambda x: 1 if x >= i else 0)
      result = {}
        
      # Metrics
      for metric_name, metric_func in metrics.items():
        if metric_name == "Youden's J statistic":
          tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
          sensitivity = tp/(tp + fn)
          specificity = tn/(tn + fp)
          print(metric_name, metric_func(y_true, y_pred))
          result[metric_name] = metric_func(y_true, y_pred)
        else:
          print(metric_name, metric_func(y_true, y_pred))
          result[metric_name] = metric_func(y_true, y_pred)
      results[i] = result

    for metric_name in metrics.keys():
      values = [result[metric_name] for result in results.values()]
      linestyle = "-" if col == "Model_1_1" else "--"
      plt.plot(thresholds, values, linestyle=linestyle, label = f"{metric_name} ({col})")
      max_value = max(values)
      max_threshold = thresholds[values.index(max_value)]
      plt.plot(max_threshold, max_value, "bo")

  plt.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
  plt.xlabel("Threshold")
  plt.ylabel("Value")
  plt.title("Metrics Comparison")
  plt.show()

metrics_comp(pd.read_csv("KM-03-2.csv"))

# Invisible metric (accuracy/balanced accuracy)
input("Press any key to start")
def metrics_comp_invis(df_new_func):
  df = df_new_func
  cols = ["Model_1_1", "Model_2_1"]
  metrics = {"Accuracy": accuracy_score,
             "Precision": precision_score,
             "Balanced Accuracy": balanced_accuracy_score}

  fig, ax = plt.subplots(figsize=(10, 6))
  for col in cols:
    print(df[col].name)
    thresholds = np.linspace(0.1, 0.99, 10)
    results = {}
    for i in thresholds:
      print(i)
      df = df_new_func
      y_true = df["GT"]
      y_pred = df[col]
      y_pred = y_pred.apply(lambda x: 1 if x >= i else 0)
      result = {}
        
      # Metrics
      for metric_name, metric_func in metrics.items():
        print(metric_name, metric_func(y_true, y_pred))
        result[metric_name] = metric_func(y_true, y_pred)
      results[i] = result

    for metric_name in metrics.keys():
      values = [result[metric_name] for result in results.values()]
      linestyle = "-" if col == "Model_1_1" else "--"
      plt.plot(thresholds, values, linestyle=linestyle, label=f"{metric_name} ({col})")
      max_value = max(values)
      max_threshold = thresholds[values.index(max_value)]
      plt.plot(max_threshold, max_value, "bo")

  plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
  plt.xlabel("Threshold")
  plt.ylabel("Value")
  plt.title("Metrics Comparison")
  plt.show()

metrics_comp_invis(pd.read_csv("KM-03-2.csv"))

input_ = input("Press any key to start")
def class1_comp(df_new_func):
  cols = ["Model_1_1", "Model_2_1"]
  for col in cols:
    df = df_new_func
    y_true = df["GT"]
    y_pred = df[col]
    thresholds = np.linspace(0.1, 0.99, 10)
    class1_count_metrics = []
    metrics = {"Accuracy": accuracy_score,
               "Precision": precision_score,
               "Recall": recall_score,
               "F-score": f1_score,
               "MCC": matthews_corrcoef,
               "Balanced Accuracy": balanced_accuracy_score,
               "AUC for Precision-Recall Curve": average_precision_score,
               "AUC for Receiver Operating Characteristic Curve": roc_auc_score}

    for t in thresholds:
      y_pred_bin = y_pred.apply(lambda x: 1 if x >= t else 0)
      class1_count = np.sum(y_pred_bin)
      metric_values = [metric_func(y_true, y_pred_bin) for metric_func in metrics.values()]
      class1_count_metrics.append((class1_count, metric_values))

    fig, ax = plt.subplots(figsize = (10, 6))
    for i, metric_name in enumerate(metrics.keys()):
      metric_values = [x[1][i] for x in class1_count_metrics]
      plt.plot(metric_values, [x[0] for x in class1_count_metrics], label = metric_name)
      max_metric_value = max(metric_values)
      max_class1_count = class1_count_metrics[metric_values.index(max_metric_value)][0]
      plt.plot(max_metric_value, max_class1_count, "bo") 
      plt.axvline(x = max_metric_value, color = "black", linestyle = "--", linewidth = 1)  

    plt.xlabel("Metric value")
    plt.ylabel("1-class quantity")
    plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    plt.title("1-class Quantity and Metric Value Comparison")
    plt.show()

class1_comp(pd.read_csv("KM-03-2.csv"))

input_ = input("Press any key to start")

def class1_comp_invis(df_new_func):
  cols = ["Model_1_1", "Model_2_1"]
  for col in cols:
    df = df_new_func
    y_true = df["GT"]
    y_pred = df[col]
    thresholds = np.linspace(0.1, 0.99, 10)
    class1_count_metrics = []
    metrics = {"Accuracy": accuracy_score,
               "Precision": precision_score,
               "Balanced Accuracy": balanced_accuracy_score}

    for t in thresholds:
      y_pred_bin = y_pred.apply(lambda x: 1 if x >= t else 0)
      class1_count = np.sum(y_pred_bin)
      metric_values = [metric_func(y_true, y_pred_bin) for metric_func in metrics.values()]
      class1_count_metrics.append((class1_count, metric_values))

    fig, ax = plt.subplots(figsize = (10, 6))
    for i, metric_name in enumerate(metrics.keys()):
      metric_values = [x[1][i] for x in class1_count_metrics]
      plt.plot(metric_values, [x[0] for x in class1_count_metrics], label = metric_name)
      max_metric_value = max(metric_values)
      max_class1_count = class1_count_metrics[metric_values.index(max_metric_value)][0]
      plt.plot(max_metric_value, max_class1_count, "bo") 
      plt.axvline(x = max_metric_value, color = "black", linestyle = "--", linewidth = 1) 

    plt.xlabel("Metric value")
    plt.ylabel("1-class quantity")
    plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    plt.title("1-class Quantity and Metric Value Comparison")
    plt.show()

class1_comp_invis(pd.read_csv("KM-03-2.csv"))

input_ = input("Press any key to start")

def roc_pr_curves(df):
  cols = ["Model_1_1", "Model_2_1"]
  fig, ax = plt.subplots(figsize = (10, 6))
  for i, col in enumerate(cols):
    y_true = df["GT"]
    y_pred = df[col]

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    f1_scores = 2 * (precision * recall)/(precision + recall)
    optimal_threshold = thresholds[f1_scores.argmax()]

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    j_scores = tpr - fpr
    optimal_threshold_roc = thresholds[j_scores.argmax()]

    if i == 0:
      color = "red"
      label = "Model 1"
      linestyle = "-"
    else:
      color = "blue"
      label = "Model 2"
      linestyle = "--"

    ax.plot(recall, precision, label=label + f" (AUC-PR = {pr_auc:.2f})", color = color, linestyle = linestyle)
    ax.scatter(recall[f1_scores.argmax()], precision[f1_scores.argmax()], marker = "o", color = color, label = f"Optimal Threshold(PR) ({col}) = {optimal_threshold:.2f}")
    ax.fill_between(recall, precision, alpha=0.1, color = color)
    ax.plot(fpr, tpr, label = label + f" (AUC-ROC = {roc_auc:.2f})", color = color, linestyle = linestyle)
    ax.scatter(fpr[j_scores.argmax()], tpr[j_scores.argmax()], marker = "o", color = color, label = f"Optimal Threshold(ROC) ({col}) = {optimal_threshold_roc:.2f}")
    ax.fill_between(fpr, tpr, alpha = 0.1, color = color)

  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1])
  ax.set_xlabel("Recall/False Positive Rate")
  ax.set_ylabel("Precision/True Positive Rate")
  ax.set_title("PR and ROC Curves")
  ax.legend(loc = "lower left")
  ax.grid(True)
  plt.show()

roc_pr_curves(pd.read_csv("KM-03-2.csv"))

# New df
input_ = input("Press any key to start")

birthday = "05-03" 
date_obj = datetime.strptime(birthday, "%d-%m")
day = date_obj.day
num_to_del = 50 + 5 * (day % 9) # 5%9 outputs 5
perc_to_del = num_to_del/100
df = pd.read_csv("KM-03-2.csv")
num_to_remove = int(perc_to_del * len(df[df["GT"] == 1]))
idx_to_remove = df[df["GT"] == 1].sample(num_to_remove).index
df.drop(idx_to_remove, inplace=True)
df_new = df

input_ = input("Press any key to start")

print("The number of deleted 1-class values: ", perc_to_del)
value_1 = len(df[df["GT"] == 1])
value_0 = len(df[df["GT"] == 0])
print(value_1)
print(value_0)

balance_ratio = value_1/value_0
print(balance_ratio)

input_ = input("Press any key to start")
metrics_comp(df_new)

input_ = input("Press any key to start")
metrics_comp_invis(df_new)

input_ = input("Press any key to start")
class1_comp(df_new)

class1_comp_invis(df_new)

input_ = input("Press any key to start")
roc_pr_curves(df_new)