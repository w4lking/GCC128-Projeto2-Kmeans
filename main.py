import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans as KMeans_sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


class KMeans_Do_Zero:
    # Adicionamos n_init=10 como padrão, igual ao scikit-learn
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init  # NOVO: Número de inicializações
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None # NOVO: Para guardar a melhor inércia

    def fit(self, X):
        # Variável para guardar a menor inércia encontrada
        best_inertia = np.inf

        # NOVO: Loop principal que executa o algoritmo n_init vezes
        for _ in range(self.n_init):
            
            # --- Início de uma única execução do K-means ---
            
            # 1. Inicializa centróides aleatórios.
            # A semente foi removida daqui para garantir que cada execução seja diferente.
            idx = np.random.choice(len(X), self.n_clusters, replace=False)
            centroids = X[idx, :]

            for _ in range(self.max_iter):
                # 2. Atribui os pontos aos clusters mais próximos
                distancias = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distancias, axis=1)

                # 3. Recalcula os centróides
                novos_centroids = np.array([
                    X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
                    for i in range(self.n_clusters)
                ])

                # 4. Verifica a convergência
                if np.all(np.abs(novos_centroids - centroids) < self.tol):
                    break
                centroids = novos_centroids
            
            # --- Fim da execução única ---

            # 5. NOVO: Calcula a inércia para esta execução
            current_inertia = 0
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    # Soma das distâncias ao quadrado
                    current_inertia += np.sum((cluster_points - centroids[i]) ** 2)

            # 6. NOVO: Compara com a melhor inércia e guarda o resultado se for melhor
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                self.centroids = centroids
                self.labels_ = labels
                self.inertia_ = best_inertia

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

    # do zero (agora rodando 10x por padrão)
    start_time = time.time()
    # A classe agora faz o trabalho pesado de rodar 10x por dentro
    meu_kmeans = KMeans_Do_Zero(n_clusters=k).fit(X)
    end_time = time.time()
    tempo_do_zero = end_time - start_time
    sil_do_zero = silhouette_score(X, meu_kmeans.labels_)
    print(f"Do zero (n_init=10): silhouette = {sil_do_zero:.4f}, tempo = {tempo_do_zero:.4f}s")

    if sil_do_zero > melhor_silhouette:
        melhor_silhouette = sil_do_zero
        melhor_k = k
        melhor_modelo = meu_kmeans

    # sklearn (usando a configuração padrão robusta)
    start_time = time.time()
    kmeans_sklearn = KMeans_sklearn(n_clusters=k, random_state=42).fit(X)
    end_time = time.time()
    tempo_sklearn = end_time - start_time
    sil_sklearn = silhouette_score(X, kmeans_sklearn.labels_)
    print(f"Sklearn (padrão):    silhouette = {sil_sklearn:.4f}, tempo = {tempo_sklearn:.4f}s")

print(f"\n>>> Melhor K encontrado = {melhor_k} com silhouette = {melhor_silhouette:.4f}\n")


# ====== PCA para visualização com o melhor K ======
# (O restante do código para plotagem permanece o mesmo)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)
centroids_pca_2d = pca_2d.transform(melhor_modelo.centroids)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=melhor_modelo.labels_, cmap="viridis", s=40)
plt.scatter(centroids_pca_2d[:, 0], centroids_pca_2d[:, 1], c="red", marker="X", s=200)
plt.title(f"Clusters (KMeans Do Zero) com PCA (2 componentes) - K={melhor_k}")
plt.show()