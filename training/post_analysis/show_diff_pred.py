import pickle, os, shutil

# Put misclassified ones together
f1 = open('res18-norotate_wonglist.pkl','rb')
wrong_1_list = pickle.load(f1)
print('wrong all', len(wrong_1_list))

f2 = open('res18-examplar-norotate-270_wonglist.pkl','rb')
wrong_2_list = pickle.load(f2)
print('wrong examplar', len(wrong_2_list))
#
diff = list(set(wrong_1_list) - set(wrong_2_list))  # This is the fixed caused by reducing training  +
# TODO: conclusion, more view diversed category corrected less examples when reduce training set
# More diversified ones can fix more original wrong by reduce training set

diff2 = list(set(wrong_2_list) -set(wrong_1_list))  # This is the misclassification caused by reducing training  -
# TODO: conclusion, more view diversed category can more wrong when reduce viewpoints
# TODO: less been removed, less accuracy decrease

print( len(diff),len(diff2))
exit(0)

target_path = '/home/mcz/2020Spring/imagenet_diff2_fixed'
# target_path = '/home/mcz/2020Spring/imagenet_noremove_diff'
#
# for each in wrong_1_list:
#     source = each
#     category = each.split('/')[-2]
#     print('cat', category)
#     tar_final = os.path.join(target_path, category)
#     os.makedirs(tar_final, exist_ok=True)
#     shutil.copy(source, os.path.join(tar_final, each.split('/')[-1]))



for each in diff2:
    source = each
    category = each.split('/')[-2]
    # print('cat', category)
    tar_final = os.path.join(target_path, category)
    os.makedirs(tar_final, exist_ok=True)
    shutil.copy(source, os.path.join(tar_final, each.split('/')[-1]))



traindir_nov = '/local/rcs/mcz/2020Spring/cut_imgnet-3.5/train'
traindir_cluster = '/local/rcs/mcz/2020Spring/cut_imgnet-3.5/train_clustered-3.5'
filelist = os.listdir(target_path)
x_list = []
y_list = []
for each in filelist:
    subfolder_list = os.listdir(os.path.join(target_path, each))
    # train_sub_list = os.listdir(os.path.join(traindir_nov, each))
    train_sub_list = os.listdir(os.path.join(traindir_cluster, each))
    print(each, 'train', len(train_sub_list), 'test wrong diff', len(subfolder_list))
    x_list.append(len(train_sub_list))
    y_list.append(len(subfolder_list))


import numpy as np
import matplotlib.pyplot as plt
x = np.asarray(x_list)
y = np.asarray(y_list)

from scipy.stats import linregress
# m,b = np.ployfit(x, y, 1)
m,b,r_value, p_value, std_err = linregress(x,y)
print(m)
print('r_value, p_value, std_err',r_value, p_value, std_err)
plt.plot(x, y, 'o')
plt.plot(x, m*x+b)
# plt.show()
plt.savefig('regress.jpg')

# print('df', diff)


#TODO: making test set viewpiont and calculate if any viewpoints easier to classifier
# We need to do view point for each test individually, copy to make a test 50 times larger than original test,
# THen calculate the accuracy for each viewpoint and show the results
# 50 is too much, just do random 3 greedy and see if if it is significantly different, if not, then fine
# And show those viewpoints with higher accuracy. See if there's canonical class.



