import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.interpolate import UnivariateSpline
import glob, os
def do_plot(file, label_, color_):
  raw_data = np.genfromtxt(file, dtype=float, skip_header =2);

  raw_x = raw_data[:,1];
  raw_y = raw_data[:,0];

  x = np.unique(raw_x);
  y = np.zeros( x.size);
  y_err = np.zeros(x.size);
  for i in range(x.size):
      y[i] = np.average(raw_y[raw_x==x[i]])
      y_err[i] = np.std(raw_y[raw_x==x[i]])/np.sqrt(np.sum(raw_x==x[i]));


  spl = UnivariateSpline(x, y)
  x_spl = np.linspace(np.min(x), np.max(x), 100)

  plt.plot(x_spl, np.interp(x_spl, x, y), color = color_)
  plt.hold(True);
  plt.errorbar(x, y, yerr=y_err, fmt='o', color = color_, label = label_)
  plt.hold(True);
  plt.legend(loc="upper left", fontsize=12, title= "Image size")
  plt.xticks(x, [str(int(xi)) for xi in x])


file_list = glob.glob('image_size_*__kernel_eps_size_21.txt');
file_list = np.array(file_list)
im_sizes = [[int(s) for s in file.split('_') if s.isdigit()][0] for file in file_list]
file_list = file_list[np.argsort(im_sizes)[::-1]]
jet = cm = plt.get_cmap('Paired')
cNorm  = colors.Normalize(vmin=0, vmax=(len(file_list)-1))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
print(colors)
i = 0;
plt.figure(figsize=(12, 9));
for file in file_list:
  print(file)
  label = [s for s in file.split('_') if s.isdigit()][0]



  do_plot(file, label, scalarMap.to_rgba(i))
  i = 1+i

plt.title('Runtime on GPUXXX',  fontsize=23)
plt.xlabel('Spatial kernel length', fontsize=18)
plt.ylabel('Runtime (ms)', fontsize=18)

plt.show()
