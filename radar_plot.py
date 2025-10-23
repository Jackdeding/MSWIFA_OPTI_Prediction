import numpy as np
import matplotlib.pyplot as plt

class Radar(object):
    def __init__(self, figure, title, labels, epoch, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]
        self.n = 5
        self.angles = np.arange(0, 360, 360.0 / self.n)
        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]
        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels = title, fontsize=16, va="center", ha="center", weight = 'bold')
        self.ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=1)
        self.ax.xaxis.grid(True, color='grey', linestyle='-', linewidth=1)
        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)
        for ax, angle, label, i in zip(self.axes, self.angles, labels, epoch):
            ax.set_rgrids(i[1:], angle=angle, labels=label, fontsize=13, zorder=10)
            ax.spines['polar'].set_visible(False)
            ax.set_rlim(i[0], i[3])
            self.epoch = epoch

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        # 根据epo值调整变换公式
        values[0] = values[0]
        values[1] = ((values[1] - 0.8) / (1 - 0.8)) * (50 - 8) + 8
        values[2] = ((values[2] - 50) / (200 - 50)) * (50 - 8) + 8
        values[3] = ((values[3] - 0.7) / (1 - 0.7)) * (50 - 8) + 8
        values[4] = ((values[4] - 50) / (110 - 50)) * (50 - 8) + 8

        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)
        print(values[0],values[1],values[2],values[3],values[4])
if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 8))
    tit = ['MAE', 'R$^2$', 'SMAPE', 'EVS', 'RMSE']

    lab = [
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
        list(('', '', '')),
        ]

    epo = [
        (8, 22, 36, 50),
        (0.8, 0.87, 0.94, 1),
        (50, 100, 150, 200),
        (0.7, 0.8, 0.9, 1),
        (50, 70, 90, 110),
            ]
    plt.rcParams['font.family'] = 'Arial'
    radar = Radar(fig, tit, lab, epo)


    radar.plot([13.579,0.978,148.834,0.978,44.799], '-', lw=4, color='tab:red', alpha=1, label='DT', marker='o', markersize=10)
    radar.plot([47.416,0.898,146.793,0.902,97.066], '-', lw=4, color='tab:blue', alpha=1, label='MLP', marker='s', markersize=10)
    radar.plot([16.801,0.938,155.81,0.938,75.949], '-', lw=4, color='tab:brown', alpha=1, label='SVR', marker='^', markersize=10)
    radar.plot([37.169,0.907,145.972,0.907,93.031], '-', lw=4, color='tab:green', alpha=1, label='RF', marker='D', markersize=10)
    radar.plot([8.847,0.985,70.290,0.985,37.034], '-', lw=4, color='tab:orange', alpha=1, label='GB', marker='x', markersize=10)
    radar.plot([12.049,0.984,153.540,0.984,38.536], '-', lw=4, color='tab:pink', alpha=1, label='XGB', marker='<', markersize=10)

    legend = radar.ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize = 15)
    plt.setp(legend.get_texts(), fontweight='bold')
    plt.show()