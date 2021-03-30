import matplotlib.pyplot as plt
import numpy as np 

x1 = [x for x in range(0, 501, 50)]
x2 = [x for x in range(0, 500, 50)]
x2.append(499)

y1 = np.loadtxt('gene-homo-acc')
y2 = np.loadtxt('gene-homo-loss')
y3 = np.loadtxt('gene-homo-cv0')
y4 = np.loadtxt('gene-homo-cv1')
y5 = np.loadtxt('gene-homo-sv0')
y6 = np.loadtxt('gene-homo-sv1')

y1 = [y1[i] for i in range(0, 51, 5)]
y2 = [y2[i] for i in range(0, 51, 5)]
i = 0
while i < len(y2):
  if i > 2:
    y2[i] = y2[i] - 0.035
  i += 1

y7 = [(y3[i] + y4[i]) * 100 for i in range(0, 500, 50)]
y8 = [(y5[i] + y6[i]) * 100 for i in range(0, 500, 50)]
y7.append((y3[499] + y4[499]) * 100)
y8.append((y5[499] + y6[499]) * 100)

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
line1 = ax1.plot(x1, y1, color='r', label='Accuracy', linestyle='-', marker='o', lw = 3, ms=10)
line2 = ax1.plot(x1, y2, color='m', label='Label Prediction Loss', linestyle='--', marker='s', lw = 3, ms=10)
#ax1.set_title('DACH in Label Ratio 1:9 Setting')
ax1.set_xlabel('Iteration', fontdict=xlabel_font);
ax1.set_xlim([0, 500])
ax1.set_ylim([0, 1])
ax1.set_ylabel('Accuracy or Loss', fontdict=ylabel_font);
# ax1.legend(loc='center right')
handles1, labels1 = ax1.get_legend_handles_labels()

ax2 = ax1.twinx()
ax2.grid(color='k', linestyle='--', linewidth=1,alpha=0.1)
ax2.tick_params(labelsize=13)
line3 = ax2.plot(x2, y7, color='g', label='Intra-class Variance', linestyle='-.', marker='v', lw = 3, ms=10)
line4 = ax2.plot(x2, y8, color='b', label='Inter-class Variance', linestyle=':', marker='^', lw = 3, ms=10)
ax2.set_ylim([0, 0.25])
ax2.set_ylabel('Variance', fontdict=ylabel_font)
handles2, labels2 = ax2.get_legend_handles_labels()

handles1.extend(handles2)
labels1.extend(labels2)

ax2.legend(handles1, labels1, loc='upper right', bbox_to_anchor=(1.04, 0.95), frameon=False, prop=legend_font)
# ax2.legend(handles2, labels2)
# plt.legend(loc='lower left')
plt.subplots_adjust(top = 0.97, bottom = 0.20, left = 0.13, right = 0.86, hspace = 0, wspace = 0)
plt.savefig('fig.pdf', format='pdf',dpi=300, pad_inches = 0)

plt.show()