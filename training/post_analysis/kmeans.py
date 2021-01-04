from sklearn.cluster import KMeans
import os
import numpy as np
import pickle
import shutil

# fea = np.load('fea1.npy')
# print('fea', fea)
with open('fea1.pkl', 'rb') as f:
    fea = pickle.load(f)

# print('fea', fea['path'])
number_of_clusters=30
km = KMeans(n_clusters=number_of_clusters, random_state=0).fit(fea['fea'])

# progressive SVM examplar (greedy) is also good

print(km.labels_)

from MulticoreTSNE import MulticoreTSNE as TSNE
tsne = TSNE(n_jobs=4, perplexity=20)
embeddings = tsne.fit_transform(fea['fea'])
from matplotlib import pyplot as plt
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]

plt.scatter(vis_x, vis_y, c=km.labels_, cmap=plt.cm.get_cmap("jet", number_of_clusters))
plt.colorbar(ticks=range(number_of_clusters))
plt.clim(-0.5, number_of_clusters-0.5)
plt.show()



new_dataset_save_path = '/mnt/md0/2020Spring/invariant_imagenet/train'
os.makedirs(new_dataset_save_path, exist_ok=True)

for i, each in enumerate(km.labels_):
    path_name = fea['path'][i][0]
    path_split = path_name.split('/')
    class_folder = path_split[-3]
    print('class', class_folder)
    # exit(0)
    cluster_path = os.path.join(os.path.join(new_dataset_save_path, class_folder), str(each))
    os.makedirs(cluster_path, exist_ok=True)

    shutil.copy(fea['path'][i][0].replace('/temp',''), os.path.join(cluster_path, path_split[-1]))

np.save(os.path.join(cluster_path, ))

# THEN move the diff labels into diff folders to check