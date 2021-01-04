import os 
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder_path = os.getcwd() + "/train_examplar"

class_name = {}

for folder_name in listdir(folder_path):
	if folder_name.endswith('.DS_Store'):
		continue
	class_name[folder_name] = {}
	for view_points in listdir(folder_path + '/' + folder_name):
		if view_points.endswith('.DS_Store'):
			continue
		curr_path = folder_path + '/' + folder_name + '/' + view_points
		class_name[folder_name][int(view_points)] = len(os.listdir(curr_path))
	
classes = list(class_name.keys())
classes.sort()

df = pd.DataFrame(class_name)
df = df.sort_index(1)

z=df.values

fig, ax = plt.subplots()
plt.imshow(z, origin='lower', aspect='auto')

ax.set_xticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
plt.setp(ax.get_xticklabels(), rotation=75, ha="right", rotation_mode="anchor")

plt.colorbar()
plt.show()