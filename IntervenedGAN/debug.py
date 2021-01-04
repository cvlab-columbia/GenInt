import os, random, shutil

path = '/proj/vondrick/mcz/GANdata/GANspaceGen/setting_50_16'
path_tar = '/local/vondrick/cz/GANdata/setting_50_16_sub'

for each in os.listdir(path):
    cat = os.path.join(path, each)
    tar_cat = os.path.join(path_tar, each)
    # if len(os.listdir(cat))<10:
    #     print(each, len(os.listdir(cat)))
    print("cat ", each)
    for c in os.listdir(cat):
        p_ins = os.path.join(cat, c)
        p_ins_tar = os.path.join(tar_cat, c)

        comp_list = os.listdir(p_ins)

        com_select = random.sample(comp_list, 2)

        shutil.copytree(os.path.join(p_ins, com_select[0]), os.path.join(p_ins_tar, com_select[0]))
        shutil.copytree(os.path.join(p_ins, com_select[1]), os.path.join(p_ins_tar, com_select[1]))


