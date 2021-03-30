import matplotlib.pyplot as plt
import numpy as np 

x1 = [x for x in range(0, 501, 50)]
x2 = [x for x in range(0, 500, 50)]
x2.append(499)

y1 = np.loadtxt('gene-homo-acc')
y2 = np.loadtxt('gene-homo-lploss')
y3 = np.loadtxt('gene-homo-dcloss')

y1 = [y1[i] for i in range(0, 51, 5)]
y2 = [y2[i] for i in range(0, 51, 5)]

y4 = [-y3[i] for i in range(0, 500, 50)]
y4.append(-y3[499])

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

fig, ax1 = plt.subplots(figsize=(5, 2.35))
ax1.grid(color='k', linestyle='--', linewidth=1,alpha=0.1)
ax1.tick_params(labelsize=13)
ax1.plot(x1, y1, color='r', label='Accuracy', linestyle='-', marker='o', lw = 3, ms=10)

# ax1.set_title('UDA in Label Ratio 1:9 Setting')
ax1.set_xlabel('Iteration', fontdict=xlabel_font);
ax1.set_xlim([0, 500])
ax1.set_ylim([0, 1])
ax1.set_ylabel('Accuracy', fontdict=ylabel_font);
# ax1.legend(loc='upper center')
handles1, labels1 = ax1.get_legend_handles_labels()

ax2 = ax1.twinx()
ax2.grid(color='k', linestyle='--', linewidth=1, alpha=0.1)
ax2.tick_params(labelsize=13)
ax2.plot(x1, y2, color='m', label='Label Prediction Loss', linestyle='--', marker='s', lw = 3, ms=10)
ax2.plot(x2, y4, color='g', label='Domain Classifier Loss', linestyle='-.', marker='v', lw = 3, ms=10)
ax2.set_ylim([-50, 50])
ax2.set_ylabel('Loss', fontdict=ylabel_font)
# ax2.legend(loc='lower left')
handles2, labels2 = ax2.get_legend_handles_labels()

handles1.extend(handles2)
labels1.extend(labels2)

ax2.legend(handles1, labels1, loc='lower left', bbox_to_anchor=(-0.02, -0.06), prop=legend_font, frameon=False)
plt.subplots_adjust(top = 0.97, bottom = 0.20, left = 0.13, right = 0.86, hspace = 0, wspace = 0)
plt.savefig('fig.pdf', format='pdf',dpi=300, pad_inches = 0)

plt.show()