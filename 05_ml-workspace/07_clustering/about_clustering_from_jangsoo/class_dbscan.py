import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm

class class_dbscan:
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
            self.X, self.y = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=0)
        elif data_type == 'moons':
            self.X, self.y = make_moons(n_samples=300, noise=0.1, random_state=0)
        elif data_type == 'circles':
            self.X, self.y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=0)
        else:
            self.X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=0)
            

            
    def dbscan_main(self, eps=0.2, min_samples=5):
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.clusters = dbscan.fit_predict(self.X)     
        self.K = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)

        self.plot_dbscan_plot()


    def plot_dbscan_plot(self, title='dbscan result'):
        fig, ax = plt.subplots()
        # 색상을 클러스터 수에 맞게 자동으로 생성
        colors = cm.tab10.colors  # tab10은 최대 10개의 색상을 제공
        for k in range(self.K):
            color = colors[k % len(colors)]  # 클러스터 수가 10을 넘을 경우 색상을 반복 사용
            points = self.X[self.clusters == k]
            ax.scatter(points[:, 0], points[:, 1], s=30, c=[color], label=f'Cluster {k+1}', alpha=0.6)
        
        # 노이즈 포인트를 회색으로 표시
        noise_points = self.X[self.clusters == -1]
        ax.scatter(noise_points[:, 0], noise_points[:, 1], s=30, c='black', label='Noise', alpha=0.6)
        
        ax.set_title(title)
        ax.legend()
        plt.show()


    def update_centroids(self):
        # 각 군집의 중심점을 계산합니다.
        self.centroids = np.array([self.X[self.clusters == i].mean(axis=0) for i in range(self.K)])

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
        # 고유한 클러스터 레이블 확인
        unique_labels = set(self.clusters)
        print(f"Unique labels: {unique_labels}")
        
        # 클러스터 개수가 2 이상인지 확인
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        print(f"Number of clusters (excluding noise): {n_clusters}")
        
        # 클러스터가 두 개 이상일 때만 silhouette_score 계산
        if n_clusters > 1 and len(self.X) == len(self.clusters):
            silhouette = silhouette_score(self.X, self.clusters)
        else:
            print("Silhouette score를 계산할 수 없습니다. 클러스터가 하나뿐이거나 데이터 불일치가 있습니다.")
            return 0

        return silhouette

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
            # cluster_points가 비어 있는지 확인
            if cluster_points.size > 0:
                intra_cluster_dist = np.max(cdist(cluster_points, [self.centroids[cluster_id]], metric='euclidean'))
            else:
                intra_cluster_dist = 0  # 비어 있는 경우 적절한 기본값 설정 (예: 0)

            if intra_cluster_dist > max_intra_cluster_dist:
                max_intra_cluster_dist = intra_cluster_dist

        # Dunn Index 계산
        dunn_index = min_inter_cluster_dist / max_intra_cluster_dist
        return dunn_index

    def evaluate(self):
        self.sse = self.calculate_sse()
        self.silhouette = self.calculate_silhouette_score()
        self.dunn_index = self.calculate_dunn_index()
        
        print(f"\nSum of Squared Errors (SSE): {self.sse}")
        print(f"Silhouette Score: {self.silhouette}")
        print(f"Dunn Index: {self.dunn_index}")



if __name__ == '__main__':
    cb = class_dbscan(interaction_flag=False)

    cb.init_dataset(data_type='moons')
    cb.dbscan_main(eps=0.16, min_samples=5)


    # cb.init_dataset(data_type='circles')
    # cb.dbscan_main(eps=0.2, min_samples=5)

    cb.evaluate()
    
