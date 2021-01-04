import numpy, os, shutil, pickle
'''Using the existing viewpoint split folder, and the example wise wronglist, to create viewpoint accuracy'''
def gen_acc_viewpoint(wrong_list, viewpoints_path, target_path):
    '''

    :param wrong_list: the wrong list of model prediction
    :param viewpoints_path: the folder of the model prediction, where
    :param target_path:
    :return:
    '''
    f1 = open(wrong_list, 'rb')
    wrong_1_list = pickle.load(f1)

    # PUt the wong list in a data structure using dict: {catogory: list of wrong files}
    hirarchy_dict = {}
    for each in wrong_1_list:
        each_s = each.split('/')
        category = each_s[-2]
        filename = each_s[-1]

        if category not in hirarchy_dict:
            hirarchy_dict[category] = []

        hirarchy_dict[category].append(filename)

    # print(hirarchy_dict)
    # exit(0)
    category_list = os.listdir(viewpoints_path)
    result_list = []
    x_list = []
    y_list = []
    for each_cat in category_list:
        if each_cat in hirarchy_dict: # Error exist for that class
            cat_path = os.path.join(viewpoints_path, each_cat)

            view_list = os.listdir(cat_path)
            # Calculate accuracy for each test view point calculated
            for each_view in view_list:
                view_path = os.path.join(cat_path, each_view)
                list_examples = os.listdir(view_path)
                total_file = len(list_examples)
                wrong_num = 0
                # print(hirarchy_dict[each_cat])
                # print(list_examples)
                for each_example in list_examples:
                    each_example = each_example.replace('0a-query_','')
                    # try:
                    if each_example in hirarchy_dict[each_cat]:
                        wrong_num += 1
                        # print('find one')
                    # except:
                    #     print(hirarchy_dict[each_cat], each_example)

                acc = 1-  wrong_num * 1.0 / total_file
                result_list.append('{}/{}/{}/{}'.format(each_cat, each_view, acc, total_file))
                x_list.append(total_file)
                y_list.append(acc)

                # Save the viewpoint with the corresponding accuracy
                try:
                    shutil.copytree(view_path, os.path.join(os.path.join(target_path, each_cat), str(acc)+'___' + each_view))
                except:
                    pass

    import numpy as np
    import matplotlib.pyplot as plt
    x = np.asarray(x_list)
    y = np.asarray(y_list)

    from scipy.stats import linregress
    # m,b = np.ployfit(x, y, 1)
    m, b, r_value, p_value, std_err = linregress(x, y)
    print(m)
    print('r_value, p_value, std_err', r_value, p_value, std_err)
    plt.plot(x, y, 'o')
    plt.plot(x, m * x + b)
    # plt.show()
    plt.savefig('regress_cano.jpg')

    # print('res', result_list)
    return result_list

# def gen_visualization(result_list,viewpoints_path):


# TODO: one conclusion is, certain viewpoint perform better than others, simply because they have more examples in that
# viewpoint


if __name__ == '__main__':
    # gen_acc_viewpoint('res18-norotate_wonglist.pkl', '/local/rcs/mcz/2020Spring/test_view/C+4', '/local/rcs/mcz/2020Spring/test_view/C+4Canonical')
    gen_acc_viewpoint('res18-examplar-norotate-270_wonglist.pkl', '/local/rcs/mcz/2020Spring/test_view/C+4', '/local/rcs/mcz/2020Spring/test_view/C+4Canonical-remove-overlap')
