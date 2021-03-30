import matplotlib.pyplot as plt
import numpy as np 

x1 = [x for x in range(1, 7)]

y1 = [0.904, 0.949, 0.951, 0.950, 0.953, 0.959]

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
ax1.plot(x1, y1, color='r', label='DACH with Gridding Operator', linestyle='-', marker='o', lw = 3, ms=10)

# ax1.set_title('DACH with Different Layers')
ax1.set_xlabel('Domain Number', fontdict=xlabel_font);
ax1.set_xlim([1, 6])
ax1.set_ylim([0.90, 0.96])
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

ax1.legend(handles1, labels1, loc='lower right', frameon=False, prop=legend_font)

plt.subplots_adjust(top = 0.97, bottom = 0.16, left = 0.15, right = 0.98, hspace = 0, wspace = 0)
plt.savefig('fig.pdf', format='pdf',dpi=300, pad_inches = 0)

plt.show()