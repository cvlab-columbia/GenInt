import os, pickle
import numpy as np

feature_path = '/local/vondrick/cz/ImageNet/VGGfeatures/train' # new ones
feature_path = '/local/vondrick/cz/ImageNet/VGGfeatures_NoNorm/train' # new ones
            # self.cluster_path_root = '/local/vondrick/cz/cut_img-S3.5-TC5/train_clustered-C{}'.format(self.C) # old
# Now I want to split the whole set without removing overlap trains
imagenet_path = '/local/vondrick/cz/ImageNet'

topk = 300

class_list = os.listdir(feature_path)

cluster_center = {}

class_name_list = []
center_array = np.zeros((1000, 8192), dtype=np.float32)

for jj, each in enumerate(sorted(os.listdir(feature_path))):
    with open(os.path.join(feature_path, each), 'rb') as f:
        fea = pickle.load(f)['fea']

    class_name_list.append(each)
    print(jj, each)

    center = np.mean(fea, axis=0)

    center_array[jj, :] = center

    # if jj==9:
    #     break

# center_array = center_array / (np.sum(center_array**2, axis=1, keepdims=True)**0.5 + 1e-10)

similarity = np.dot(center_array, center_array.T)
print(similarity)
for jj in range(similarity.shape[0]):
    subarray = similarity[jj, :]
    top_k_index = subarray.argsort()[-topk-1:-1][::-1]
    print(top_k_index)
    cluster_center[class_name_list[jj]] = top_k_index

saveall = {'cluster_topk_index': cluster_center, 'name_list': class_name_list}


with open('cluster_nneighbor.pkl', 'wb') as f:
    pickle.dump(saveall, f)












