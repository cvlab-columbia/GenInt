import json, os
import shutil

def get_imagenet_overlap():
    with open('preprocessing/obj2imgnet_id.txt') as f:
        dict_obj2imagenet_id = json.load(f)

    overlapping_list= []
    for each in dict_obj2imagenet_id.keys():
        overlapping_list.extend(dict_obj2imagenet_id[each])

    all_list = [i for i in range(1000)]

    non_overlapping = list(set(all_list) - set(overlapping_list))

    print('non-overlapping', len(non_overlapping))

    return overlapping_list, non_overlapping

def gen_imagenet_overlap_data(input_path, output_path):

    in_categories = os.listdir(input_path)
    in_categories.sort()
    category2id = {filename: fileintkey for fileintkey, filename in enumerate(in_categories)}
    id2category = {fileintkey: filename for fileintkey, filename in enumerate(in_categories)}

    with open('preprocessing/obj2imgnet_id.txt') as f:
        dict_obj2imagenet_id = json.load(f)

    for key in dict_obj2imagenet_id.keys():
        id_content = dict_obj2imagenet_id[key]
        name_g_list = []
        name_whole = ''
        for efo in id_content:
            name_g_list.append(id2category[efo])
            name_whole = name_whole + id2category[efo]
        os.makedirs(os.path.join(output_path, name_whole), exist_ok=True)
        for eachone in name_g_list:
            filelist = os.listdir(os.path.join(input_path, eachone))
            print("copy {}".format(eachone))
            for eachfile in filelist:
                shutil.copy(os.path.join(os.path.join(input_path, eachone), eachfile),
                            os.path.join(os.path.join(output_path, name_whole), eachfile))





# self.overlap_subset = overlap_subset
# self.grouping = {}
# if overlap_subset:
#     with open('preprocessing/obj2imgnet_id.txt') as f:
#         self.dict_obj2imagenet_id = json.load(f)
#
#     for key in self.dict_obj2imagenet_id.keys():
#         id_content = self.dict_obj2imagenet_id[key]
#         # convert id to name:
#         name_g_list = []
#         for efo in id_content:
#             name_g_list.append(self.id2category[efo])
#         self.grouping[name_g_list[0]] = name_g_list
#         # Here, we construct a dictionary, that map the beginning categories to the other categories.







