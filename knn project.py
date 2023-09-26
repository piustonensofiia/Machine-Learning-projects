import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, ShuffleSplit, RandomizedSearchCV
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, roc_auc_score, precision_recall_curve, roc_curve, auc

# Відкрити та зчитати наданий файл з даними
input_ = input("Press any key to continue")
df = pd.read_csv("dataset2_l4.txt", sep = ",")

# Визначити та вивести кількість записів
input_ = input("Press any key to continue")
print(df.shape[0])
print(df.shape[1])

# Вивести атрибути набору даних
input_ = input("Press any key to continue")
print(df.columns)

# З’ясувати збалансованість набору даних
input_ = input("Press any key to continue")
df.head()
class_counts = df["Class"].value_counts() 
print(class_counts)
is_balanced = (class_counts.max() - class_counts.min()) <= 1
if is_balanced:
    print("Balanced")
else:
    print("Aren't balanced")

# Отримати двадцять варіантів перемішування набору даних та розділення його на навчальну (тренувальну) 
# та тестову вибірки, використовуючи функцію ShuffleSplit. 
# Сформувати начальну та тестові вибірки на основі обраного користувачем варіанту
input_ = input("Press any key to continue")
shuffled = ShuffleSplit(train_size=0.8, test_size=0.2, n_splits=20, random_state=42)
list_of_shuffle_split_options = []
for train_index, test_index in shuffled.split(df):
    print("train:", train_index, "test:", test_index)
    list_of_shuffle_split_options.append((train_index, test_index))

choice = int(input("Choose ShuffleSplitted from 1 to 20: "))
print(list_of_shuffle_split_options[choice - 1])

# Використовуючи функцію KNeighborsClassifier бібліотеки scikit-learn, збудувати
# класифікаційну модель на основі методу k найближчих сусідів 
# (кількість сусідів обрати самостійно, вибір аргументувати) та навчити її на тренувальній вибірці, 
# вважаючи, що цільова характеристика визначається стовпчиком Class, а всі інші виступають в ролі 
# вихідних аргументів
input_ = input("Press any key to continue")
train_sample_indexes = list_of_shuffle_split_options[choice - 1][0]
X_train = [df.drop("Class", axis=1).iloc[index].tolist() for index in train_sample_indexes]
y_train = [df["Class"].iloc[index] for index in train_sample_indexes]

# k-fold cross-validation
# param_grid = {"n_neighbors": range(1, 30)}
# knn = KNeighborsClassifier()
# knn_cv = GridSearchCV(knn, param_grid, cv=5)
# knn_cv.fit(X_train, y_train)
# print(knn_cv.best_params_)

# stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5)
param_grid = {"n_neighbors": range(2, 30)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=skf)
knn_cv.fit(X_train, y_train)
print(knn_cv.best_params_)

# random search
# param_dist = {"n_neighbors": randint(1, 30)}
# knn = KNeighborsClassifier()
# knn_cv = RandomizedSearchCV(knn, param_dist, cv=5, n_iter=10)
# knn_cv.fit(X_train, y_train)
# print(knn_cv.best_params_)

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train, y_train)

# Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки. 
# Представити результати роботи моделі на тестовій вибірці графічно
input_ = input("Press any key to continue")
def metrics_count(y_train_pred, y_test_pred):
    metrics = {"Accuracy": accuracy_score,
             "Precision": precision_score,
             "Recall": recall_score,
             "F-score": f1_score,
             "MCC": matthews_corrcoef,
             "Accuracy(Bal)": balanced_accuracy_score}
    results = {}

    for metric_name, metric_func in metrics.items():
      if metric_name in ["Accuracy", "MCC", "Accuracy(Bal)"]: 
        train_score = metric_func(y_train, y_train_pred)
        test_score = metric_func(y_test, y_test_pred)
        print(metric_name, "train/test", round(train_score, 3), "/", round(test_score, 3))
      else:
        train_score = metric_func(y_train, y_train_pred, average="macro")
        test_score = metric_func(y_test, y_test_pred, average="macro")
        print(metric_name, "train/test", round(train_score, 3), "/", round(test_score, 3))

test_sample_indexes = list_of_shuffle_split_options[choice - 1][1]
X_test = [df.drop("Class", axis=1).iloc[index].tolist() for index in test_sample_indexes]
y_test = [df["Class"].iloc[index] for index in test_sample_indexes]
metrics_count(KNN.predict(X_train), KNN.predict(X_test))

#------------------------------------------------------------------
cm = confusion_matrix(y_test, KNN.predict(X_test))
sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.show()

# Обрати алгоритм KDTree та з’ясувати вплив розміру листа (від 20 до 200 з кроком 5) на 
# результати класифікації. Результати представити графічно
characteristics = []
for size in range(20, 201, 5):
  start_time = time.time()
  KNN = KNeighborsClassifier(algorithm="kd_tree", leaf_size=size)
  KNN.fit(X_train, y_train)
  y_test_predict = KNN.predict(X_test)
  time_delta = time.time() - start_time
  count_was_predicted = sum(y_test == y_test_predict)
  characteristics.append((size, time_delta, count_was_predicted))

labels, values1, values2 = zip(*characteristics)
plt.bar(labels, values1, color="pink")
plt.title("Leaf size/Time")
plt.show()

plt.bar(labels, values2, color="pink")
plt.title("Leaf size/predicted correctly")
plt.show()

# add-ly
input_ = input("Press any key to continue")

df1 = df[["F1", "F2", "F3", "F4", "F5", "F9", "Class"]]
X = df1.drop("Class", axis = 1) 
y = df1["Class"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5)
param_grid = {"n_neighbors": range(2, 30)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=skf)
knn_cv.fit(X_train, y_train)
print(knn_cv.best_params_)

KNN = KNeighborsClassifier(n_neighbors=2)
KNN.fit(X_train, y_train)

metrics_count(KNN.predict(X_train), KNN.predict(X_test))

cm = confusion_matrix(y_test, KNN.predict(X_test))
sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.show()