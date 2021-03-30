import matplotlib.pyplot as plt
import numpy as np 

x1 = [x for x in range(1, 6)]
x2 = [x for x in range(1, 6)]
x3 = [x for x in range(1, 6)]
x4 = [x for x in range(1, 6)]
x5 = [x for x in range(1, 6)]
x6 = [x for x in range(1, 6)]

y1 = [0.922, 0.923, 0.920, 0.918, 0.917]
y2 = [0.896, 0.872, 0.822, 0.785, 0.742]
y3 = [0.891, 0.910, 0.807, 0.677, 0.580]
y4 = [0.843, 0.819, 0.799, 0.762, 0.752]
y5 = [0.883, 0.874, 0.852, 0.838, 0.828]
y6 = [0.901, 0.905, 0.879, 0.868, 0.826]

xlabel_font = {'family' : 'Times New Roman',
               'weight' : 'normal',
               'size'   : 13
              }
ylabel_font = {'family' : 'Times New Roman',
               'weight' : 'normal',
               'size'   : 13
              }
legend_font = {'family' : 'Times New Roman',
               'weight' : 'normal',
               'size'   : 13
              }

fig, ax1 = plt.subplots(figsize=(5, 3))
ax1.grid(color='k', linestyle='--', linewidth=1,alpha=0.1)
ax1.tick_params(labelsize=13)
ax1.plot(x1, y1, color='r', label='DACH', linestyle='-', marker='o', lw = 3, ms=10)
ax1.plot(x2, y2, color='m', label='LatentDA', linestyle='--', marker='s', lw = 3, ms=10)
ax1.plot(x3, y3, color='g', label='GRL', linestyle='-.', marker='v', lw = 3, ms=10)
ax1.plot(x4, y4, color='c', label='DRCN', linestyle=':', marker='^', lw = 3, ms=10)
ax1.plot(x5, y5, color='b', label='DAN', linestyle=(0, (3, 1, 1, 1, 1, 1)), marker='<', lw = 3, ms=10)
ax1.plot(x6, y6, color='k', label='DDC', linestyle=(0, (5, 2, 2, 2, 2, 2)), marker='>', lw = 3, ms=10)

# ax1.set_title('Accuracy with Different Methods')
ax1.set_xlabel('Class Ratio', fontdict=xlabel_font);
# ax1.set_xlim([1, 5])
ax1.set_ylim([0.5, 1.0])
ax1.set_ylabel('Accuracy', fontdict=ylabel_font);
# ax1.legend(loc='upper center')
handles1, labels1 = ax1.get_legend_handles_labels()

# ax2 = ax1.twinx()
# ax2.plot(x1, y2, color='m', label='Label Prediction Loss', marker='s')
# ax2.plot(x2, y4, color='b', label='Domain Classifier Loss', marker='v')
# ax2.set_ylim([-50, 50])
# ax2.set_ylabel('Loss')
# ax2.legend(loc='lower left')
# handles2, labels2 = ax2.get_legend_handles_labels()

# handles1.extend(handles2)
# labels1.extend(labels2)

ax1.legend(handles1, labels1, loc='lower left', bbox_to_anchor=(-0.026, -0.07), frameon=False, prop=legend_font)
plt.xticks(x1, ['5:5', '6:4', '7:3', '8:2', '9:1'])

plt.subplots_adjust(top = 0.97, bottom = 0.16, left = 0.13, right = 0.98, hspace = 0, wspace = 0)
plt.savefig('fig.pdf', format='pdf', dpi=300, pad_inches = 0)
plt.show()