import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, completeness_score, adjusted_rand_score, homogeneity_score, silhouette_score, pairwise_distances, davies_bouldin_score, v_measure_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.spatial import distance
import sys

# Відкрити та зчитати наданий файл з даними
input_ = input("Press any key to continue")
df = pd.read_csv("dataset2_l4.txt", sep = ",")

# Визначити та вивести кількість записів
input_ = input("Press any key to continue")
print(df.shape[0])
print(df.shape[1])

# Видалити атрибут Class
input_ = input("Press any key to continue")
df.drop(["Class"], axis=1, inplace=True)
df.head()

# Вивести атрибути, що залишилися
input_ = input("Press any key to continue")
attributes = df.columns.tolist()
print(attributes)

# Використовуючи функцію KMeans бібліотеки scikit-learn, виконати 
# розбиття набору даних на кластери з випадковою початковою 
# ініціалізацією і вивести координати центрів кластерів.
input_ = input("Press any key to continue")
kmeans = KMeans(init="random")
kmeans.fit(df)
cluster_centers = kmeans.cluster_centers_
for center in cluster_centers:
  print(center)

# elbow method
input_ = input("Press any key to continue")
def elbow(df):
  np.random.seed(9) 
  inertias = []
  amount = []
  for num_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=num_clusters, init="random")
    kmeans.fit(df)
    inertias.append(kmeans.inertia_)
    amount.append(num_clusters)
  plt.plot(amount, inertias, marker="o", color="black")
  plt.xlabel("Number of Clusters")
  plt.ylabel("Inertia")
  plt.title("Elbow Method")
  plt.show()
  diff = np.diff(inertias)
  diff2 = np.diff(diff)
  elbow_point_index = np.where(diff2 < 0)[0][0] + 2  
  optimal_num_clusters = amount[elbow_point_index]
  print("Optimal number of clusters:", optimal_num_clusters)

elbow(df)

# average silhouette method
input_ = input("Press any key to continue")
def average_silhouette(df):
  np.random.seed(9) 
  silhouette_scores = []
  num_clusters_list = []
  for num_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=num_clusters, init="random")
    kmeans.fit(df)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df, labels)
    silhouette_scores.append(silhouette_avg)
    num_clusters_list.append(num_clusters)

  plt.plot(num_clusters_list, silhouette_scores, marker="o", color="black")
  plt.xlabel("Number of Clusters")
  plt.ylabel("Average Silhouette Score")
  plt.title("Average Silhouette Method")
  plt.show()

  optimal_num_clusters = num_clusters_list[np.argmax(silhouette_scores)]
  print("Optimal number of clusters:", optimal_num_clusters)

average_silhouette(df)

# prediction  strength  method 
input_ = input("Press any key to continue")
def get_closest_centroid(obs, centroids):
  min_distance = sys.float_info.max
  min_centroid = 0
  for c in centroids:
    dist = distance.euclidean(obs, c)
    if dist < min_distance:
      min_distance = dist
      min_centroid = c

  return min_centroid

def get_prediction_strength(k, train_centroids, x_test, test_labels):
  n_test = len(x_test)
  D = np.zeros(shape=(n_test, n_test))
  for x1, l1, c1 in zip(x_test, test_labels, list(range(n_test))):
    for x2, l2, c2 in zip(x_test, test_labels, list(range(n_test))):
      if tuple(x1) != tuple(x2):
        if tuple(get_closest_centroid(x1, train_centroids)) == tuple(get_closest_centroid(x2, train_centroids)):
          D[c1, c2] = 1.0
  ss = []
  for j in range(k):
    s = 0
    examples_j = x_test[test_labels == j, :].tolist()
    n_examples_j = len(examples_j)
    for x1, l1, c1 in zip(x_test, test_labels, list(range(n_test))):
      for x2, l2, c2 in zip(x_test, test_labels, list(range(n_test))):
        if tuple(x1) != tuple(x2) and l1 == l2 and l1 == j:
          s += D[c1, c2]
    ss.append(s / (n_examples_j * (n_examples_j - 1)))
  prediction_strength = min(ss)
  return prediction_strength

def prediction_strength_method(): 
  X = df.to_numpy()
  X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
  strengths = []
  for k in range(1, 5):
    model_train = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_train)
    model_test = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_test)
    pred_str = get_prediction_strength(k, model_train.cluster_centers_, X_test, model_test.labels_)
    strengths.append(pred_str)

  _, ax = plt.subplots()
  ax.plot(range(1, 5), strengths, "-o", color="black")
  ax.axhline(y=0.8, c="red")
  ax.set(title="Determining the optimal number of clusters",
       xlabel="number of clusters",
       ylabel="prediction strength")
  plt.show()

choice = int(input("Do you want to start PSM 0-no/1-yes"))
if choice == 1:
  prediction_strength_method()

# За раніш обраної кількості кластерів багаторазово проведіть 
# кластеризацію методом k-середніх, використовуючи для початкової 
# ініціалізації метод k-means++
input_ = input("Press any key to continue")
def metrics_k_means(df, df_with_target, n):
  inertias = []
  silhouette_scores = []
  davies_bouldin_scores = []
  predicted_labels = []
  completeness_ = []
  homogeneity_ = []
  v_measure_ = []
  ami_ = []
  ri_ = []

  for i in range(10):
    kmeans = KMeans(n_clusters=n, init="k-means++", n_init=1)
    kmeans.fit(df)
    labels = kmeans.labels_
    predicted_labels.append(labels)
    inertia = kmeans.inertia_
    inertias.append(inertia)
    silhouette_avg = silhouette_score(df, labels)
    silhouette_scores.append(silhouette_avg)
    davies_bouldin_score_ = davies_bouldin_score(df, labels)
    davies_bouldin_scores.append(davies_bouldin_score_)
    completeness = completeness_score(df_with_target["Class"], predicted_labels[i])
    completeness_.append(completeness)
    homogeneity = homogeneity_score(df_with_target["Class"], predicted_labels[i])
    homogeneity_.append(homogeneity)
    v_measure = v_measure_score(df_with_target["Class"], predicted_labels[i])
    v_measure_.append(v_measure)
    ami = adjusted_mutual_info_score(df_with_target["Class"], predicted_labels[i])
    ami_.append(ami)
    ri = adjusted_rand_score(df_with_target["Class"],  predicted_labels[i])
    ri_.append(ri)

  for i in range(10):
    print("Model", i+1)
    print("Inertia:", inertias[i])
    print("Silhouette Score:", silhouette_scores[i])
    print("Davies-Bouldin Score:", davies_bouldin_scores[i])
    print("----------------------------------------------")
    print("Completeness:", completeness_[i])
    print("Homogeneity:", homogeneity_[i])
    print("V-Measure:", v_measure_[i])
    print("AMI", ami_[i])
    print("RI", ri_[i])
    print("**********************************************")

  print("\n")
  print("The best model due to the Inertia metric", np.argmin(inertias)+1, "TOP-3: ", np.argsort(inertias)[:3]+1)
  print("The best model due to the Silhouette metric", np.argmax(silhouette_scores)+1, "TOP-3: ", np.argsort(silhouette_scores)[::-1][:3]+1)
  print("The best model due to the Davies-Bouldin Index metric", np.argmin(davies_bouldin_scores)+1, "TOP-3: ", np.argsort(davies_bouldin_scores)[:3]+1)
  print("The best model due to the Completeness metric:", np.argmax(completeness_) + 1, "TOP-3:", np.argsort(completeness_)[::-1][:3] + 1)
  print("The best model due to the Homogeneity metric:", np.argmax(homogeneity_) + 1, "TOP-3:", np.argsort(homogeneity_)[::-1][:3] + 1)
  print("The best model due to the V-Measure metric:", np.argmax(v_measure_) + 1, "TOP-3:", np.argsort(v_measure_)[::-1][:3] + 1)
  print("The best model due to the AMI metric:", np.argmax(ami_) + 1, "TOP-3:", np.argsort(ami_)[::-1][:3] + 1)
  print("The best model due to the RI metric:", np.argmax(ri_) + 1, "TOP-3:", np.argsort(ri_)[::-1][:3] + 1)

df_1 = pd.read_csv("dataset2_l4.txt", sep = ",")
metrics_k_means(df, df_1, 3)

# Використовуючи функцію AgglomerativeClustering бібліотеки scikit-
# learn, виконати розбиття набору даних на кластери. Кількість кластерів 
# обрати такою ж самою, як і в попередньому методі. Вивести 
# координати центрів кластерів
input_ = input("Press any key to continue")
def metrics_agglomerative(df, df_with_target, n):
  agglomerative = AgglomerativeClustering(n_clusters=n)
  agglomerative.fit(df)

  print("Silhouette Score:", silhouette_score(df, agglomerative.labels_))
  print("Davies-Bouldin Score:", davies_bouldin_score(df, agglomerative.labels_))
  print("Completeness:", completeness_score(df_with_target["Class"], agglomerative.labels_))
  print("Homogeneity:", homogeneity_score(df_with_target["Class"], agglomerative.labels_))
  print("V-Measure:", v_measure_score(df_with_target["Class"], agglomerative.labels_))
  print("AMI", adjusted_mutual_info_score(df_with_target["Class"], agglomerative.labels_))
  print("RI", adjusted_rand_score(df_with_target["Class"], agglomerative.labels_))

  cluster_centers = []
  for label in range(3):
      cluster_points = df[agglomerative.labels_ == label]
      center = cluster_points.mean(axis=0)
      cluster_centers.append(center)
      print(center)

df_1 = pd.read_csv("dataset2_l4.txt", sep = ",")
metrics_agglomerative(df, df_1, 3)

# DATA INVESTIGATION
input_ = input("Press any key to continue")
df = pd.read_csv("dataset2_l4.txt", sep = ",")
label_encoder = LabelEncoder()
df["Class"] = label_encoder.fit_transform(df["Class"])
print("Сorrelation F1/Class:", np.corrcoef(df["F1"], df["Class"])[0, 1])
print("Сorrelation F2/Class:", np.corrcoef(df["F2"], df["Class"])[0, 1])
print("Сorrelation F3/Class:", np.corrcoef(df["F3"], df["Class"])[0, 1])
print("Сorrelation F4/Class:", np.corrcoef(df["F4"], df["Class"])[0, 1])
print("Сorrelation F5/Class:", np.corrcoef(df["F5"], df["Class"])[0, 1])
print("Сorrelation F6/Class:", np.corrcoef(df["F6"], df["Class"])[0, 1])
print("Сorrelation F7/Class:", np.corrcoef(df["F7"], df["Class"])[0, 1])
print("Сorrelation F8/Class:", np.corrcoef(df["F8"], df["Class"])[0, 1])
print("Сorrelation F9/Class:", np.corrcoef(df["F9"], df["Class"])[0, 1])
print("Сorrelation F10/Class:", np.corrcoef(df["F10"], df["Class"])[0, 1])

input_ = input("Press any key to continue")
df_1 = df[["F5", "F9"]]
df_2 = df[["F5", "F9", "Class"]]
elbow(df_1)

input_ = input("Press any key to continue")
metrics_k_means(df_1, df_2, 6)
metrics_agglomerative(df_1, df_2, 6)

input_ = input("Press any key to continue")
pca = PCA(n_components = 2)
pca_data = pca.fit_transform(df)
elbow(pca_data)

input_ = input("Press any key to continue")
kmeans = KMeans(n_clusters=6, init="k-means++", n_init=1)
kmeans.fit(pca_data)
labels = kmeans.labels_
metrics_k_means(pca_data, df, 6)
metrics_agglomerative(pca_data, df, 6)