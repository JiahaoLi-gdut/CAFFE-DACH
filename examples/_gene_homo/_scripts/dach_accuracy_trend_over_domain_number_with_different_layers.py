import matplotlib.pyplot as plt
import numpy as np 

x1 = [x for x in range(1, 7)]
x2 = [x for x in range(1, 7)]

y1 = [0.904, 0.949, 0.951, 0.950, 0.953, 0.959]
y2 = [0.512, 0.497, 0.505, 0.507, 0.497, 0.520]

xlabel_font = {'family' : 'Times New Roman',
               'weight' : 'normal',
               'size'   : 23
              }
ylabel_font = {'family' : 'Times New Roman',
               'weight' : 'normal',
               'size'   : 23
              }
legend_font = {'family' : 'Times New Roman',
               'weight' : 'normal',
               'size'   : 23
              }

fig, ax1 = plt.subplots(figsize=(4, 3))
ax1.grid(color='k', linestyle='--', linewidth=1,alpha=0.1)
ax1.tick_params(labelsize=23)
ax1.plot(x1, y1, color='r', label='Gridding', linestyle='-', marker='o', lw = 6, ms=23)
ax1.plot(x2, y2, color='b', label='Convolution', linestyle='--', marker='s', lw = 6, ms=23)

# ax1.set_title('DACH with Different Layers')
ax1.set_xlabel('Source Number', fontdict=xlabel_font);
ax1.set_xticks([1,2,3,4,5,6])
ax1.set_xlim([1, 6])
ax1.set_ylim([0.4, 1.0])
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

ax1.legend(handles1, labels1, loc='center right', frameon=False, prop=legend_font)

plt.subplots_adjust(top = 0.96, bottom = 0.24, left = 0.24, right = 0.98, hspace = 0, wspace = 0)
plt.savefig('fig.pdf', format='pdf',dpi=300, pad_inches = 0)

plt.show()