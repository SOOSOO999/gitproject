import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

import matplotlib.cm as cm

class class_kmeans:
    def __init__(self, k=3, max_iters=10, interaction_flag=True):
        self.K = k
        self.max_iters = max_iters
        
        self.interaction_flag = interaction_flag
        
        self.X, self.y = None, None
        self.centroids = None
        self.clusters = None
        
        self.SSE, self.silhouette, self.dunn_index = None, None, None
        
    def init_dataset(self, data_type='blobs'):
        if data_type == 'blobs':
            self.X, self.y = make_blobs(n_samples=300, centers=self.K, cluster_std=0.7, random_state=None)
        elif data_type == 'moons':
            self.X, self.y = make_moons(n_samples=300, noise=0.1, random_state=0)
        elif data_type == 'circles':
            self.X, self.y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=0)
        else:
            self.X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=0)
            
        

    def initialize_centroids(self):
        # 데이터셋 X에서 무작위로 k개의 초기 중심점을 선택
        indices = np.random.choice(self.X.shape[0], self.K, replace=False)
        self.centroids = self.X[indices]
        
        # 초기화된 centroid를 보여줌
        fig, ax = plt.subplots()
        colors = cm.tab10.colors  # 최대 10개의 색상을 지원하는 colormap
        for k in range(self.K):
            ax.scatter(self.X[:, 0], self.X[:, 1], s=30, c='gray', alpha=0.5)
        color = colors[k % len(colors)]    
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], s=400, c=[color], marker='X', edgecolor='k', linewidth=2, alpha=0.9)
        ax.set_title('Initial centroids')
        plt.show()
        
        if self.interaction_flag:
            input("Press Enter to start iterations...")



    def assign_clusters(self):
        clusters = []
        for point in self.X:
            distances = np.linalg.norm(point - self.centroids, axis=1)
            cluster = np.argmin(distances)
            clusters.append(cluster)
        self.clusters = np.array(clusters)
        
    def update_centroids(self):
        self.centroids = np.array([self.X[self.clusters == i].mean(axis=0) for i in range(self.K)])

    def has_converged(self, old_centroids):
        tolerance = 1e-4

        distances = np.linalg.norm(self.centroids - old_centroids, axis=1)
        return np.all(distances < tolerance)

    def kmeans_main(self):

        self.initialize_centroids()
        
        for i in range(self.max_iters):
            self.assign_clusters()
            
            self.plot_kmeans_plot(title=f'Iteration {i + 1} - After Assignment')
            
            if self.interaction_flag:
                input(f"Iteration {i + 1}: data reassignment done...")            
            
            old_centroids = self.centroids
            self.update_centroids()
            
            self.plot_kmeans_plot(title=f'Iteration {i + 1} - After Centroid Update')
            if self.interaction_flag:
                input(f"Iteration {i + 1}: centroid update done...")            
            
            if self.has_converged(old_centroids):
                print(f"K-means converged after {i+1} iterations")
                break


    def plot_kmeans_plot(self, title='kmeans result'):
        fig, ax = plt.subplots()
        colors = cm.tab10.colors  # 최대 10개의 색상을 지원하는 colormap
        for k in range(self.K):
            points = self.X[self.clusters == k]
            color = colors[k % len(colors)]
            ax.scatter(points[:, 0], points[:, 1], s=30, c=[color], label=f'Cluster {k+1}', alpha=0.6)
        
        # 중심점의 색상을 각 클러스터 색상에 맞추어 개별적으로 설정
        for idx, centroid in enumerate(self.centroids):
            ax.scatter(centroid[0], centroid[1], s=400, c=[colors[idx % len(colors)]], 
                       marker='X', edgecolor='k', linewidth=2, alpha=0.9)
            
        ax.set_title(title)
        ax.legend()
        plt.show()


    def calculate_sse(self):
        sse = 0.0
        for cluster_id in range(self.K):
            cluster_points = self.X[self.clusters == cluster_id]
            if self.centroids is not None and len(self.centroids) > 0:
                centroid = self.centroids[cluster_id]
            else:
                self.update_centroids()
                centroid = self.centroids[cluster_id]
                
            sse += np.sum((cluster_points - centroid) ** 2)
        return sse
    
    

    def calculate_silhouette_score(self):
        return silhouette_score(self.X, self.clusters)


    def calculate_dunn_index(self):
        # 클러스터 간 최단 거리 (inter-cluster distance)
        min_inter_cluster_dist = np.inf
        for i in range(self.K):
            for j in range(i + 1, self.K):
                inter_cluster_dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                if inter_cluster_dist < min_inter_cluster_dist:
                    min_inter_cluster_dist = inter_cluster_dist

        # 클러스터 내 최대 거리 (intra-cluster distance)
        max_intra_cluster_dist = 0
        for cluster_id in range(self.K):
            cluster_points = self.X[self.clusters == cluster_id]
            intra_cluster_dist = np.max(cdist(cluster_points, [self.centroids[cluster_id]], metric='euclidean'))
            if intra_cluster_dist > max_intra_cluster_dist:
                max_intra_cluster_dist = intra_cluster_dist

        # Dunn Index 계산
        dunn_index = min_inter_cluster_dist / max_intra_cluster_dist
        return dunn_index



    def evaluate(self):
        """
        클러스터링 품질을 평가하고 결과를 출력
        """
        self.sse = self.calculate_sse()
        self.silhouette = self.calculate_silhouette_score()
        self.dunn_index = self.calculate_dunn_index()
        
        print("\nSum of Squared Errors (SSE):", self.sse)
        print("Silhouette Score:", self.silhouette)
        print("Dunn Index:", self.dunn_index)



if __name__ == '__main__':
    ck = class_kmeans(k=5, interaction_flag=False)
    ck.init_dataset(data_type='blobs')
    ck.kmeans_main()
    ck.evaluate()

    
