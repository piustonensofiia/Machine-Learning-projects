import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sns

# Відкрити та зчитати наданий файл з даними.
input_ = input("Press any key to continue")
df = pd.read_csv("dataset_2.txt", sep = ",", header = None)
df.columns = ["№", "DateTime", "Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Actual Value"]

# Визначити та вивести кількість записів та кількість полів у завантаженому наборі даних.
input_ = input("Press any key to continue")
print(df.shape[0])
print(df.shape[1])

# Вивести перші 10 записів набору даних.
input_ = input("Press any key to continue")
df.head(10)

# Розділити набір даних на навчальну (тренувальну) та тестову вибірки.
input_ = input("Press any key to continue")
df = df.drop(df.columns[:2], axis = 1)

X = df.drop("Actual Value", axis = 1) 
y = df["Actual Value"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
df.head()

# Використовуючи відповідні функції бібліотеки scikit-learn, збудувати класифікаційну модель дерева прийняття рішень глибини 5 та навчити її на тренувальній вибірці, вважаючи, що в наданому наборі даних цільова характеристика визначається останнім стовпчиком, а всі інші (окрім двох перших) виступають в ролі вихідних аргументів.
input_ = input("Press any key to continue")
clf = DecisionTreeClassifier(max_depth = 5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Представити графічно побудоване дерево за допомогою бібліотеки graphviz.
input_ = input("Press any key to continue")
dot_data = export_graphviz(clf, feature_names = X.columns)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки. 
input_ = input("Press any key to continue")

def metrics_count(y_train_pred, y_test_pred, color_):
    metrics = {"Accuracy": accuracy_score,
             "Precision": precision_score,
             "Recall": recall_score,
             "F-score": f1_score,
             "MCC": matthews_corrcoef,
             "Accuracy(Bal)": balanced_accuracy_score,
             "AUC for PRC": average_precision_score,
             "AUC for ROCC": roc_auc_score}
    results = {}

    for metric_name, metric_func in metrics.items():
        train_score = metric_func(y_train, y_train_pred)
        test_score = metric_func(y_test, y_test_pred)
        results[metric_name + "_train"] = train_score
        results[metric_name + "_test"] = test_score
        print(metric_name, "train/test", train_score, test_score)
        
    sns.lineplot(x=list(metrics.keys()), y=[round(results[metric_name + "_train"], 3) for metric_name in metrics.keys()], color=color_, linestyle="--", label="Train")
    sns.lineplot(x=list(metrics.keys()), y=[round(results[metric_name + "_test"], 3) for metric_name in metrics.keys()], color=color_, label="Test")
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)

metrics_count(clf.predict(X_train), clf.predict(X_test), "green")

# Представити результати роботи моделі на тестовій вибірці графічно.
input_ = input("Press any key to continue")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.show()

# Порівняти результати, отриманні при застосуванні різних критеріїв розщеплення: інформаційний приріст на основі ентропії чи неоднорідності Джині.
input_ = input("Press any key to continue")
clf = DecisionTreeClassifier(max_depth = 5, criterion = "entropy")
clf.fit(X_train, y_train)
metrics_count(clf.predict(X_train), clf.predict(X_test), "green")
print("------------")
clf_1 = DecisionTreeClassifier(max_depth = 5, criterion = "gini")
clf_1.fit(X_train, y_train)
metrics_count(clf_1.predict(X_train), clf_1.predict(X_test), "red")
plt.show()

# З'ясувати вплив максимальної кількості листів та мінімальної кількості елементів в листі дерева на результати класифікації. Результати представити графічно.
input_ = input("Press any key to continue")
max_leaf_nodes = [4, 8, 12, 16, 20]
min_samples_leaf = [1, 2, 3, 4, 5]

metrics = {"Accuracy": accuracy_score,
           "Precision": precision_score,
           "Recall": recall_score,
           "F-score": f1_score,
           "MCC": matthews_corrcoef,
           "Accuracy(Bal)": balanced_accuracy_score,
           "AUC for PRC": average_precision_score,
           "AUC for ROCC": roc_auc_score}

results = {metric: {"train": [], "test": []} for metric in metrics.keys()}

fig, axes = plt.subplots(len(metrics), figsize=(12, 20), sharex=True)
for i, (metric, scores) in enumerate(results.items()):
    for max_leaf in max_leaf_nodes:
        for min_leaf in min_samples_leaf:
            clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf, min_samples_leaf=min_leaf)
            clf.fit(X_train, y_train)
            train_preds = clf.predict(X_train)
            test_preds = clf.predict(X_test)
            train_score = metrics[metric](y_train, train_preds)
            test_score = metrics[metric](y_test, test_preds)
            results[metric]["train"].append(train_score)
            results[metric]["test"].append(test_score)

    axes[i].plot(range(len(results[metric]["train"])), results[metric]["train"], label=None, color="green", linestyle="--")
    axes[i].plot(range(len(results[metric]["test"])), results[metric]["test"], label=None, color="green")
    axes[i].set_ylabel(metric)
axes[-1].set_xlabel("Combination of Max Leaf Nodes and Min Samples Leaf")
plt.xticks(ticks=range(len(max_leaf_nodes) * len(min_samples_leaf)), labels=[f"{max_leaf}-{min_leaf}" for max_leaf in max_leaf_nodes for min_leaf in min_samples_leaf], rotation="vertical")
plt.suptitle("Model Performance Metrics vs. Max Leaf Nodes and Min Samples Leaf")
plt.legend(["Train", "Test"])
plt.show()

# Навести стовпчикову діаграму важливості атрибутів, які використовувалися для класифікації (див. feature_importances_). Пояснити, яким чином – на Вашу думку – цю важливість можна підрахувати.
input_ = input("Press any key to continue")
feature_importances_ = clf.feature_importances_
feature_names = df.columns.tolist()[:-1]

plt.bar(range(len(feature_importances_)), feature_importances_, color = "green")
plt.xticks(range(len(feature_importances_)), feature_names)
plt.ylabel("Importance")
plt.xlabel("Attributes")
plt.xticks(rotation = 30)
plt.show()