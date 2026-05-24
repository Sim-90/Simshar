import numpy as np
from sklearn.cluster import KMeans
from libKMCUDA import kmeans_cuda
# import sklearn.cluster.k_means_
from sklearn.cluster import k_means as km
from sklearn.cluster import kmeans_plusplus
from sklearn.utils.extmath import row_norms, squared_norm
from numpy.random import RandomState
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE

def save_clusters(centers, labels, save_path):
    """Save cluster centers and labels to a .pth file."""
    data_to_save = {
        'centers': centers,
        'labels': labels
    }
    torch.save(data_to_save, save_path)
    print(f"Cluster data saved to {save_path}")

def plot_tsne(self, weight_vector, labels, title = "t-SNE plot for clustered weights", layer_index = 0):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(weight_vector)

        plt.figure(figsize=(12,8))
        scatter = plt.scatter(tsne_results[:,0],tsne_results[:,1], c= labels,cmap='jet',alpha=0.7)
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")

        save_path = f"./sneplots/tsne_layer_{layer_index}.png"
        if save_path:
            plt.savefig(save_path, format='png',dpi=300)
            print(f"T-sne plot aved at: {save_path}")
        plt.show()

def k_means_cpu(weight_vector, n_clusters,save_path=None, seed=int(time.time())):

	kmeans_result = KMeans(n_clusters=n_clusters, init='k-means++',  random_state = seed).fit(weight_vector)
	labels = kmeans_result.labels_
	centers = kmeans_result.cluster_centers_
	weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
	for v in range(weight_vector.shape[0]):
		weight_vector_compress[v, :] = centers[labels[v], :]
	# weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
	# 	# Save clusters if save_path is provided
	if save_path:
		save_clusters(centers, labels, save_path)

	return weight_vector_compress

def k_means_gpu(weight_vector, n_clusters, verbosity=0,save_path=None, seed=int(time.time()), gpu_id=7):

	if n_clusters == 1:

		mean_sample = np.mean(weight_vector, axis=0)

		weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))

		return weight_vector

	elif weight_vector.shape[0] == n_clusters:

		return weight_vector

	elif weight_vector.shape[1] == 1:

		return k_means_cpu(weight_vector, n_clusters, seed=seed)

	else:
		# print('n_clusters', n_clusters)
		# print('weight_vector.shape',weight_vector.shape)
		# print('kmeans++ init start')
		init_centers, _  = kmeans_plusplus(X=weight_vector, n_clusters=n_clusters, x_squared_norms=row_norms(weight_vector, squared=True), random_state=RandomState(seed))
		# # print('kmeans++ init finished')
		# # print('init_centers.shape',init_centers.shape)
		centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init=init_centers, yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)

		# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="k-means++", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)

		# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="random", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
		# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="afk-mc2", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
		weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
		for v in range(weight_vector.shape[0]):
			weight_vector_compress[v, :] = centers[labels[v], :]
		# weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
	# Save clusters if save_path is provided
		if save_path:
			save_clusters(centers, labels, save_path)

		return weight_vector_compress, centers, labels

# def k_means_gpu(weight_vector, n_clusters, verbosity=0, seed=int(time.time()), gpu_id=7):
#     if n_clusters == 1:
#         mean_sample = np.mean(weight_vector, axis=0)
#         weight_vector_compress = np.tile(mean_sample, (weight_vector.shape[0], 1))
#         labels = np.zeros(weight_vector.shape[0], dtype=np.int32)
#         centers = mean_sample[np.newaxis, :]
#         return weight_vector_compress, centers, labels

#     elif weight_vector.shape[0] == n_clusters:
#         labels = np.arange(n_clusters, dtype=np.int32)
#         centers = weight_vector.copy()
#         return weight_vector, centers, labels

#     elif weight_vector.shape[1] == 1:
#         return k_means_cpu(weight_vector, n_clusters, seed=seed)

#     else:
#         # Initialize cluster centers using k-means++
#         init_centers, _ = kmeans_plusplus(
#             X=weight_vector,
#             n_clusters=n_clusters,
#             x_squared_norms=row_norms(weight_vector, squared=True),
#             random_state=RandomState(seed)
#         )
        
#         # Perform k-means clustering using CUDA
#         centers, labels = kmeans_cuda(
#             samples=weight_vector,
#             clusters=n_clusters,
#             init=init_centers,
#             yinyang_t=0,
#             seed=seed,
#             device=gpu_id,
#             verbosity=verbosity
#         )

#         # Map each vector to its cluster center
#         weight_vector_compress = np.zeros_like(weight_vector, dtype=np.float32)
#         for v in range(weight_vector.shape[0]):
#             weight_vector_compress[v, :] = centers[labels[v], :]

#         return weight_vector_compress, centers, labels


def k_means_gpu_sparsity(weight_vector, n_clusters, ratio=0.5, verbosity=0,save_path=None, seed=int(time.time()), gpu_id=0):

	# print(n_clusters)
	if ratio == 0:

		return k_means_gpu(weight_vector=weight_vector, n_clusters=n_clusters, verbosity=verbosity, seed=seed, gpu_id=gpu_id)

	if ratio == 1:

		if n_clusters == 1:

			mean_sample = np.mean(weight_vector, axis=0)

			weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))

			return weight_vector

		elif weight_vector.shape[0] == n_clusters:

			return weight_vector

		else:
			# mean_sample = np.mean(weight_vector, axis=0)
			weight_vector_1_mean = np.mean(weight_vector, axis=0)

			weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
			for v in weight_vector.shape[0]:
				weight_vector_compress[v, :] = weight_vector_1_mean

			return weight_vector_compress

	else:

		if n_clusters == 1:

			mean_sample = np.mean(weight_vector, axis=0)

			weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))

			return weight_vector

		elif weight_vector.shape[0] == n_clusters:

			return weight_vector

		elif weight_vector.shape[1] == 1:

			return k_means_sparsity(weight_vector, n_clusters, ratio, seed=seed)

		else:
			print('n_clusters', n_clusters)
			print('weight_vector.shape',weight_vector.shape)
			print('kmeans++ init start')
			num_samples = weight_vector.shape[0]
			mean_sample = np.mean(weight_vector, axis=0)

			center_cluster_index = np.argsort(np.linalg.norm(weight_vector - mean_sample, axis=1))[
								   :int(num_samples * ratio)]
			# weight_vector_1 = weight_vector[min_index, :]
			weight_vector_1_mean = np.mean(weight_vector[center_cluster_index, :], axis=0)

			remaining_cluster_index = np.asarray([i for i in np.arange(num_samples) if i not in center_cluster_index])

			weight_vector_train = weight_vector[remaining_cluster_index, :]
			# weight_vector_train = [element for i, element in enumerate(weight_vector) if i not in min_index]
			# weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))
			init_centers, _  = kmeans_plusplus(X=weight_vector_train, n_clusters=n_clusters - 1,
															x_squared_norms=row_norms(weight_vector_train,
																					  squared=True),
															random_state=RandomState(seed))
			print('kmeans++ init finished')
			print('init_centers.shape',init_centers.shape)
			centers, labels = kmeans_cuda(samples=weight_vector_train, clusters=n_clusters - 1, init=init_centers,
										  yinyang_t=0,
										  seed=seed, device=gpu_id, verbosity=verbosity)
			# print(np.unique(labels, axis=0).shape[0]+1)
			# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="k-means++", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
			# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="random", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
			# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="afk-mc2", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
			weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
			for v in center_cluster_index:
				weight_vector_compress[v, :] = weight_vector_1_mean

			for i, v in enumerate(remaining_cluster_index):
				weight_vector_compress[v, :] = centers[labels[i], :]
			# weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
			# print(np.unique(weight_vector_compress, axis=0).shape[0])
			# print(n_clusters, '\n')
			# assert np.unique(weight_vector_compress, axis=0).shape[0]==n_clusters, "cluster number mismatch"
		# Save clusters if save_path is provided
			if save_path:
				save_clusters(centers, labels, save_path)

			return weight_vector_compress,centers, labels


# def k_means_gpu_sparsity(weight_vector, n_clusters, ratio=0.5, verbosity=0, seed=int(time.time()), gpu_id=0):
#     if ratio == 0:
#         return k_means_gpu(weight_vector=weight_vector, n_clusters=n_clusters, verbosity=verbosity, seed=seed, gpu_id=gpu_id)

#     if ratio == 1:
#         if n_clusters == 1:
#             mean_sample = np.mean(weight_vector, axis=0)
#             weight_vector_compress = np.tile(mean_sample, (weight_vector.shape[0], 1))
#             labels = np.zeros(weight_vector.shape[0], dtype=np.int32)
#             centers = mean_sample[np.newaxis, :]
#             return weight_vector_compress, centers, labels

#         elif weight_vector.shape[0] == n_clusters:
#             labels = np.arange(n_clusters, dtype=np.int32)
#             centers = weight_vector.copy()
#             return weight_vector, centers, labels

#         else:
#             mean_sample = np.mean(weight_vector, axis=0)
#             weight_vector_compress = np.zeros_like(weight_vector, dtype=np.float32)
#             labels = np.zeros(weight_vector.shape[0], dtype=np.int32)
#             for v in range(weight_vector.shape[0]):
#                 weight_vector_compress[v, :] = mean_sample
#             centers = mean_sample[np.newaxis, :]
#             return weight_vector_compress, centers, labels

#     else:
#         num_samples = weight_vector.shape[0]
#         mean_sample = np.mean(weight_vector, axis=0)

#         # Select top samples closest to the mean
#         center_cluster_index = np.argsort(
#             np.linalg.norm(weight_vector - mean_sample, axis=1)
#         )[:int(num_samples * ratio)]
#         remaining_cluster_index = np.asarray([i for i in np.arange(num_samples) if i not in center_cluster_index])

#         weight_vector_train = weight_vector[remaining_cluster_index, :]
        
#         # Initialize centers for remaining clusters
#         init_centers, _ = kmeans_plusplus(
#             X=weight_vector_train,
#             n_clusters=n_clusters - 1,
#             x_squared_norms=row_norms(weight_vector_train, squared=True),
#             random_state=RandomState(seed)
#         )
        
#         # Perform k-means clustering
#         centers, labels_train = kmeans_cuda(
#             samples=weight_vector_train,
#             clusters=n_clusters - 1,
#             init=init_centers,
#             yinyang_t=0,
#             seed=seed,
#             device=gpu_id,
#             verbosity=verbosity
#         )

#         # Reconstruct compressed weight vector
#         weight_vector_compress = np.zeros_like(weight_vector, dtype=np.float32)
#         labels = np.zeros(num_samples, dtype=np.int32)

#         # Assign mean to center cluster
#         for v in center_cluster_index:
#             weight_vector_compress[v, :] = mean_sample
#             labels[v] = 0

#         # Assign remaining clusters
#         for i, v in enumerate(remaining_cluster_index):
#             weight_vector_compress[v, :] = centers[labels_train[i], :]
#             labels[v] = labels_train[i] + 1

#         # Add mean_sample to centers
#         centers = np.vstack((mean_sample[np.newaxis, :], centers))

#         return weight_vector_compress, centers, labels


def k_means_sparsity(weight_vector, n_clusters, ratio,save_path=None, seed=int(time.time())):

	num_samples = weight_vector.shape[0]
	mean_sample = np.mean(weight_vector, axis=0)

	center_cluster_index = np.argsort(np.linalg.norm(weight_vector - mean_sample, axis=1))[:int(num_samples * ratio)]
	# weight_vector_1 = weight_vector[min_index, :]
	weight_vector_1_mean = np.mean(weight_vector[center_cluster_index, :], axis=0)

	remaining_cluster_index = np.asarray([i for i in np.arange(num_samples) if i not in center_cluster_index])

	weight_vector_train = weight_vector[remaining_cluster_index, :]
	# weight_vector_train = [element for i, element in enumerate(weight_vector) if i not in min_index]
	# weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))
	# init_centers = sklearn.cluster.k_means_._k_init(X=weight_vector_train, n_clusters=n_clusters-1,
	# 												x_squared_norms=row_norms(weight_vector_train, squared=True),
	# 												random_state=RandomState(seed))
	# # # print('kmeans++ init finished')
	# # # print('init_centers.shape',init_centers.shape)
	# centers, labels = kmeans_cuda(samples=weight_vector_train, clusters=n_clusters-1, init=init_centers, yinyang_t=0,
	# 							  seed=seed, device=gpu_id, verbosity=verbosity)
	# print(np.unique(labels, axis=0).shape[0]+1)
	# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="k-means++", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
	# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="random", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
	# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="afk-mc2", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
	kmeans_result = KMeans(n_clusters=n_clusters, init='k-means++', 
						   random_state=seed).fit(weight_vector_train)
	labels = kmeans_result.labels_
	centers = kmeans_result.cluster_centers_
	weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)

	for i, v in enumerate(remaining_cluster_index):
		weight_vector_compress[v, :] = centers[labels[i], :]

	for v in center_cluster_index:
		weight_vector_compress[v, :] = weight_vector_1_mean
	# for i, v in enumerate(remaining_cluster_index):
	# 	weight_vector_compress[v, :] = centers[labels[i], :]
	# weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
	# print(np.unique(weight_vector_compress, axis=0).shape[0])
	# print(n_clusters, '\n')
	# assert np.unique(weight_vector_compress, axis=0).shape[0]==n_clusters, "cluster number mismatch"
	
	# Save clusters if save_path is provided
	if save_path:
		save_clusters(centers, labels, save_path)

	return weight_vector_compress, centers, labels
