'''blender --python /home/me/my_script.py'''
import bpy
import os
from random import randint
from math import pi
import random

try:
    bpy.ops.object.delete(use_global=False, confirm=False)
except:
    # print('Fail delete bpy.context.selected_objects', len(bpy.context.selected_objects), bpy.context.selected_objects)
    # exit(0)
    pass
light_x = randint(-7,7)
light_y = randint(-7,7)
light_z = randint(-7,7)

### get the path where the blend file is located

# outpath = '/Users/augustinecha/Downloads/test2'
# path_to_obj_dir = '/Users/augustinecha/Downloads/'
# else:
import socket
if socket.gethostname() == 'deep':
    outpath = '/mnt/md0/2020Spring/ShapeNet_output/v3'
    path_to_obj_dir = '/mnt/md0/2020Spring/ShapeNetCore.v2'
elif socket.gethostname() == 'cv04': # THIS IS not solved
    outpath = '/proj/vondrick/mcz/ShapeNetRender/v_more'
    path_to_obj_dir = '/proj/vondrick/mcz/ShapeNet/ShapeNetCore.v2'

top_k = 100
category_list = sorted(os.listdir(path_to_obj_dir))

for each_cat in category_list:
    save_category = os.path.join(outpath, each_cat)
    os.makedirs(save_category, exist_ok=False)

    cat_path = os.path.join(path_to_obj_dir, each_cat)
    examples_list = os.listdir(cat_path)

    selected_examples = random.sample(examples_list, min(top_k, len(examples_list)))
    selected_examples = sorted(selected_examples)

    for each_example in selected_examples:
        obj_path = os.path.join(os.path.join(cat_path, each_example), 'models/model_normalized.obj')

        try:
            bpy.ops.object.delete(use_global=False, confirm=False)
        except:
            print("Fail delete")
            print('Fail delete bpy.context.selected_objects', len(bpy.context.selected_objects), bpy.context.selected_objects)
            exit(0)
            pass

        # try:
        if os.path.exists(obj_path) is False:
            continue

        save_example_path = os.path.join(save_category, each_example)
        os.makedirs(save_example_path, exist_ok=True)

        bpy.ops.import_scene.obj(filepath=obj_path)

        ### Set camera and light position
        bpy.context.scene.camera.location = (1, -1, 0.75)
        resolution_X = 256
        resolution_Y = 256
        for obj in bpy.data.objects:
            print(obj.type)
            if obj.type == "CAMERA":
                bpy.context.scene.camera = obj
                bpy.context.scene.render.resolution_x = resolution_X
                bpy.context.scene.render.resolution_y = resolution_Y
            if obj.type == "LIGHT":
                obj.location = (light_x, light_y, light_z)

        #### loop through all the objects in the scene
        scene = bpy.context.scene

        x_interval = 90
        y_interval = 90
        z_interval = 90

        # for ob in bpy.context.selected_objects:
        #     xt=0
        #     yt=0
        #     zt =0
        #     img_name = each_example + '_x_' + str(xt) + '_y_' + str(yt) + '_z' + str(zt)
        #     ob.rotation_euler = (pi * xt / 180, pi * yt / 180, pi * zt / 180)
        #     scene.render.filepath = os.path.join(save_example_path, img_name)
        #     bpy.ops.render.render(write_still=True)
        # continue

        cnt_canonical_ind = 0
        print('bpy.context.selected_objects', len(bpy.context.selected_objects), bpy.context.selected_objects)
        for ob in bpy.context.selected_objects:
            for x in range(0, 360, x_interval):
                for y in range(0, 360, y_interval):
                    for z in range(0, 360, z_interval):
                        cnt_canonical_ind += 1
                        xt = x
                        yt = y
                        zt = z
                        img_name = each_example + '_c_' + str(cnt_canonical_ind) + '_x_' + str(xt) + '_y_' + str(yt) + '_z' + str(zt)
                        ob.rotation_euler = (pi * xt / 180, pi * yt / 180, pi * zt / 180)
                        scene.render.filepath = os.path.join(save_example_path, img_name)
                        bpy.ops.render.render(write_still=True)

                        for repeat in range(2):
                            # xr = random.randint(0, x_interval)
                            # yr = random.randint(0, x_interval)
                            # zr = random.randint(0, x_interval)
                            xr = random.randint(-x_interval//2, x_interval//2)
                            yr = random.randint(-x_interval//2, x_interval//2)
                            zr = random.randint(-x_interval//2, x_interval//2)
                            xt = x + xr
                            yt = y + yr
                            zt = z + zr
                            img_name = each_example + '_c_' + str(cnt_canonical_ind) +  '_x_' + str(xt) + '_y_' + str(yt) + '_z' + str(zt)
                            ob.rotation_euler = (pi * xt / 180, pi * yt / 180, pi * zt / 180)
                            scene.render.filepath = os.path.join(save_example_path, img_name)
                            bpy.ops.render.render(write_still=True)
        # except:
        #     continue



# ### get list of all files in directory
# file_list = sorted(os.listdir(path_to_obj_dir))
# os.makedirs(outpath, exist_ok=True)
#
# ### get a list of files ending in 'obj'
# obj_list = [item for item in file_list if item[-3:] == 'obj']
#
# ### loop through the strings in obj_list and add the files to the scene
# for item in obj_list:
#     path_to_file = os.path.join(path_to_obj_dir, item)
#     bpy.ops.import_scene.obj(filepath = path_to_file)
#
# ### Set camera and light position
# bpy.context.scene.camera.location = (1,-1,0.75)
# resolution_X = 256
# resolution_Y = 256
# for obj in bpy.data.objects:
#     print(obj.type)
#     if obj.type == "CAMERA":
#         bpy.context.scene.camera = obj
#         bpy.context.scene.render.resolution_x = resolution_X
#         bpy.context.scene.render.resolution_y = resolution_Y
#     if obj.type == "LIGHT":
#         obj.location = (light_x,light_y,light_z)
#
# #### loop through all the objects in the scene
# scene = bpy.context.scene
#
# x_interval = 90
# y_interval = 90
# z_interval = 90
#
# for ob in bpy.context.selected_objects:
#     for x in range(0,360, x_interval):
#         for y in range(0,360, y_interval):
#             for z in range(0,360, z_interval):
#
#                 img_name = 'x'+str(x)+'y'+str(y)+'z'+str(z)
#                 ob.rotation_euler = (pi*x/180,pi*y/180,pi*z/180)
#                 scene.render.filepath = os.path.join(outpath, img_name)
#                 bpy.ops.render.render(write_still=True)

# bpy.ops.object.delete(use_global=False, confirm=False)