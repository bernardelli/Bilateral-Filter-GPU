import numpy as np
import glob, os
import pandas as pd
from ggplot import *


file_list = glob.glob('*.txt');

gpu_name_list = [n[0:-4] for n in file_list]

#print(gpu_name_list);

#raw_data_list = [np.genfromtxt(f, names=True, skip_header=1, delimiter='\t', dtype=None) for f in file_list];
raw_data_list = [pd.read_table(f, skiprows =1, dtype=None, lineterminator ='\n', engine = 'c') for f in file_list];
#print(raw_data_list[0][:]['time_to_allocate'])
#print(raw_data_list[1].dtype)





df2 = pd.DataFrame(raw_data_list[0])
#df2 = pd.melt(df2, id_vars=['Repeating', 'Image size', 'scale_xy', 'scale_eps','kernel_xy_size', 'kernel_eps_size' ],\
#	value_vars=['cubefilling', 'convolution', 'slicing', 'time to allocate', 'time copy memory', 'time to free', 'komplete time'])


subset1 = pd.DataFrame(df2[(df2['scale_xy'] == 26) & (df2['scale_eps'] == 31) & (df2['kernel_eps_size'] == 16)]) #splice subset
#subset1 = subset1.sort(['scale_xy']);

#grouped = subset1.groupby(['Image size', 'kernel_xy_size']).agg([np.mean, np.std])
#print(grouped)



subset1['kernel_xy_size'] = subset1['kernel_xy_size'].astype('category') #.cat.codes
subset1['Image size'] = subset1['Image size'].astype('category') #.cat.codes


new = pd.DataFrame(subset1.groupby(['Image size', 'kernel_xy_size'])['convolution'].agg([np.mean]))
new2 = pd.DataFrame(subset1.groupby(['Image size', 'kernel_xy_size'])['convolution'].agg([np.std]))
print(new)
new3 = pd.concat([new, new2], axis=0).reset_index()
new3.dropna();
new3['kernel_xy_size'] = new3['kernel_xy_size'].astype('category') #.cat.codes


print(new3)

#print(df)
p = ggplot(new3,aes(x='Image size', y = 'mean', color = 'kernel_xy_size')) +\
	geom_point() +  geom_line() +\
	ggtitle("Something on" + gpu_name_list[0]) + \
	xlab("Image size") + \
	ylab("Runtime (ms)") + scale_color_brewer(type='qual')
# stat_smooth(se=False) + 

'''
instead of error bars, you could plot all the generated points with some transparency

later try geom_bar() + weight = y




levels(data1[['group']]) <- letters[1:3] 
ggplot(data1, aes(x=group, y=estimate)) + 
  geom_errorbar(aes(ymin=estimate-SE, ymax=estimate+SE), 
  colour="black", width=.1, position=pd) +
  geom_line( aes(x=as.numeric(group), y=estimate)) + 
  geom_point(position=pd, size=4)
'''




print("wtf")

p.save('plots/test.png')
