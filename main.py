import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans as KMeans_sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


class KMeans_Do_Zero:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(42)  # reprodutibilidade
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx, :]

        for _ in range(self.max_iter):
            distancias = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels_ = np.argmin(distancias, axis=1)

            novos_centroids = np.array([
                X[self.labels_ == i].mean(axis=0) if len(X[self.labels_ == i]) > 0 else self.centroids[i]
                for i in range(self.n_clusters)
            ])

            if np.all(np.abs(novos_centroids - self.centroids) < self.tol):
                break

            self.centroids = novos_centroids

        return self

    def predict(self, X):
        distancias = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distancias, axis=1)


# ====== CARREGAR DADOS ======
df = pd.read_csv("data/iris.csv")
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values

# ====== EXPERIMENTOS ======
melhor_k = None
melhor_silhouette = -1
melhor_modelo = None

for k in [3, 5]:
    print(f"\n==== K = {k} ====")

    # do zero
    start_time = time.time()
    meu_kmeans = KMeans_Do_Zero(n_clusters=k).fit(X)
    end_time = time.time()
    tempo_do_zero = end_time - start_time
    sil_do_zero = silhouette_score(X, meu_kmeans.labels_)
    print(f"Do zero: silhouette = {sil_do_zero:.4f}, tempo = {tempo_do_zero:.4f}s")

    if sil_do_zero > melhor_silhouette:
        melhor_silhouette = sil_do_zero
        melhor_k = k
        melhor_modelo = meu_kmeans

    # sklearn (apenas comparação)
    start_time = time.time()
    kmeans_sklearn = KMeans_sklearn(n_clusters=k, random_state=42).fit(X)
    end_time = time.time()
    tempo_sklearn = end_time - start_time
    sil_sklearn = silhouette_score(X, kmeans_sklearn.labels_)
    print(f"Sklearn: silhouette = {sil_sklearn:.4f}, tempo = {tempo_sklearn:.4f}s")

print(f"\n>>> Melhor K encontrado = {melhor_k} com silhouette = {melhor_silhouette:.4f}\n")


# ====== PCA para visualização com o melhor K ======

# --- 1 componente ---
pca_1d = PCA(n_components=1)
X_pca_1d = pca_1d.fit_transform(X)
centroids_pca_1d = pca_1d.transform(melhor_modelo.centroids)

plt.figure(figsize=(10, 3))
plt.scatter(X_pca_1d, np.zeros_like(X_pca_1d), c=melhor_modelo.labels_, cmap="viridis", s=40)
plt.scatter(centroids_pca_1d, np.zeros_like(centroids_pca_1d), c="red", marker="X", s=200)
plt.title(f"Clusters (KMeans Do Zero) com PCA (1 componente) - K={melhor_k}")
plt.yticks([])
plt.show()

# --- 2 componentes ---
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)
centroids_pca_2d = pca_2d.transform(melhor_modelo.centroids)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=melhor_modelo.labels_, cmap="viridis", s=40)
plt.scatter(centroids_pca_2d[:, 0], centroids_pca_2d[:, 1], c="red", marker="X", s=200)
plt.title(f"Clusters (KMeans Do Zero) com PCA (2 componentes) - K={melhor_k}")
plt.show()
