path = '/local/rcs/mcz/GANgendata/GanSpace/z-img_500_20_1_s3_lam_8.0'
import os, shutil

ip = os.path.join(path, 'z')

for jjj in range(200):
    for each in os.listdir(ip):
        p1 = os.path.join(ip, each)
        ins_list = os.listdir(p1)

        for ev in ins_list:
            if 'n' in ev:
                shutil.move(os.path.join(p1, ev), os.path.join(ip, ev))
