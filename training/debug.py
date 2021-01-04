from utils import fix_imagenet_temp
import os, pickle
import matplotlib.pyplot as plt
import shutil

# fix_imagenet_temp('/mnt/md0/ImageNet/train_debug')
#
# path = '/proj/vondrick/mcz/ImageNet-Data/ResNet152features/similarity'
#
# similarity_dict={}
# for each in os.listdir(path):
#     print(each)
#     with open(os.path.join(path, each), 'rb') as f:
#         data = pickle.load(f)
#         # dist = data['sorted_similarity']
#         # num_bins = 20
#         # for jj in range(10):
#         #     n, bins, patches = plt.hist(dist[jj][-50:], num_bins, facecolor='blue', alpha=0.5)
#         #     plt.savefig("hist{}-n.jpg".format(jj))
#         # print(dist)
#         # exit(0)
#         print(data['mapping_i2n'].keys())
#         print(data['mapping_i2n'][1])
#         exit(0)


# path='/local/vondrick/cz/GANdata/rand'
# for each in os.listdir(path):
#     os.rename(os.path.join(path, each), os.path.join(path, str(each.split('_')[0])))

# path='/local/vondrick/cz/GANdata/rand'
# sour_path = '/local/vondrick/cz/GANdata/ganrand'
#
# cnt=0
# for each in os.listdir(sour_path):
#     cat_folder = os.path.join(sour_path, each)
#     for img in os.listdir(cat_folder):
#         imgfile = os.path.join(cat_folder, img)
#         tar_file = os.path.join(os.path.join(path, each), img)
#         shutil.copy(imgfile, tar_file)
#         cnt+=1
#         if cnt%200==1:
#             print(each, cnt)


path='/local/vondrick/cz/GANdata/rand'
cnt=0
for each in os.listdir(path):
    cat = os.path.join(path, each)
    for fff in os.listdir(cat):
        if '.png' not in fff:
            fn = os.path.join(cat, fff)
            print(fn)
            # os.remove(fn)
            shutil.rmtree(fn)
    # print(len())


    # cat_folder = os.path.join(sour_path, each)
    # for img in os.listdir(cat_folder):
    #     imgfile = os.path.join(cat_folder, img)
    #     tar_file = os.path.join(os.path.join(path, each), img)
    #     shutil.copy(imgfile, tar_file)
    #     cnt+=1
    #     if cnt%200==1:
    #         print(each, cnt)
