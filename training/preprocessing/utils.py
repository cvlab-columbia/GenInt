def is_range(x,y,z, range_list):
    if x>range_list[0] and y>range_list[0] and z > range_list[0] and x<range_list[1] and y<range_list[1] and z<range_list[1]:
        return True
    else:
        return False

import shutil, os
def move_list(test_moving_list):
    for each in test_moving_list:
        try:
            folder_name = '/'.join(each[1].split('/')[:-1])
            # print(folder_name)
            os.makedirs(folder_name, exist_ok=True)
            shutil.copy(each[0], each[1])

        except:
            print(each, "not success")