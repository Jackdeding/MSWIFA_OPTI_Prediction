import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import spearmanr

violin_plot_savepath = './SHAP_plot2/dataset_statistics_violin_plot/'

def violin_plot(data_df):
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.family'] = 'Arial'
    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Ca"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='Ca', ylabel='value(%)')
    sns.boxplot(data=data_df["Ca"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Ca_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Cl"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='Cl', ylabel='value(%)')
    sns.boxplot(data=data_df["Cl"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Cl_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Si"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='Si', ylabel='value(%)')
    sns.boxplot(data=data_df["Si"], width=0.04, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Si_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Al"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='Al', ylabel='value(%)')
    sns.boxplot(data=data_df["Al"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Al_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["MC time"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='MC time', ylabel='value(h)')
    sns.boxplot(data=data_df["MC time"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'MC time_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["MC speed"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='MC speed', ylabel='value(rpm)')
    sns.boxplot(data=data_df["MC speed"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'MC speed_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["L/S"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='L/S', ylabel='value')
    sns.boxplot(data=data_df["L/S"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'L-S_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["BW/W"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='BW/W', ylabel='value')
    sns.boxplot(data=data_df["BW/W"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'BW-W_statistics.png')
    plt.close()


    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Ca-additive"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='Ca-additive', ylabel='value(%)')
    sns.boxplot(data=data_df["Ca-additive"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Ca-additive_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["P-additive"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='P-additive', ylabel='value(%)')
    sns.boxplot(data=data_df["P-additive"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'P-additive_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Ca-P additive"], inner=None, cut=0, color='b', alpha=0.45, linewidth=3).set(
        xlabel='Ca-P additive', ylabel='value(%)')
    sns.boxplot(data=data_df["Ca-P additive"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Ca-P additive_statistics.png')
    plt.close()
    #
    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Si-Al additive"], inner=None, cut=0, color='b', alpha= 0.45,linewidth=3).set(xlabel='Si-Al additive', ylabel='value(%)')
    sns.boxplot(data=data_df["Si-Al additive"], width=0.05, showcaps=True, boxprops={'facecolor': 'black','linewidth': 3}, showfliers=False, showmeans=True, meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth':2},
                medianprops={'linestyle': '-', 'linewidth': 3,'color':'white'}, whiskerprops={'linewidth': 3}, saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Si-Al additive_statistics.png')
    plt.close()


    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Leaching pH"], inner=None, cut=0, color='b', alpha=0.45, linewidth=3).set(
        xlabel='Leaching pH', ylabel='value')
    sns.boxplot(data=data_df["Leaching pH"], width=0.05, showcaps=True,
                boxprops={'facecolor': 'black', 'linewidth': 3}, showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Leaching pH_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Total metal"], inner=None, cut=0, color='b', alpha=0.35, linewidth=3).set(
        xlabel='Total metal', ylabel='value(mg/kg)')
    sns.boxplot(data=data_df["Total metal"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Total metal_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Initial concentration"], inner=None, cut=0, color='b', alpha=0.35, linewidth=3).set(
        xlabel='Initial concentration', ylabel='value(mg/l)')
    sns.boxplot(data=data_df["Initial concentration"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Initial concentration_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Ionic radius"], inner=None, cut=0, color='b', alpha=0.35, linewidth=3).set(
        xlabel='Ionic radius', ylabel='value(Å)')
    sns.boxplot(data=data_df["Ionic radius"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Ionic radius_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["Electronegativity"], inner=None, cut=0, color='b', alpha=0.35, linewidth=3).set(
        xlabel='Electronegativity', ylabel='value')
    sns.boxplot(data=data_df["Electronegativity"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'Electronegativity_statistics.png')
    plt.close()


    fig = plt.figure(dpi=600, figsize=(8, 8))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(8, 8))
    sns.set(context='paper', style='ticks', font_scale=5)
    sns.violinplot(data_df["OPTI"], inner=None, cut=0, color='b', alpha=0.35, linewidth=3).set(xlabel='OPTI',
                                                                                                    ylabel='value')
    sns.boxplot(data=data_df["OPTI"], width=0.05, showcaps=True, boxprops={'facecolor': 'black', 'linewidth': 3},
                showfliers=False, showmeans=True,
                meanprops={'marker': 'o', 'markeredgecolor': 'white', 'markerfacecolor': 'white', 'markeredgewidth': 2},
                medianprops={'linestyle': '-', 'linewidth': 3, 'color': 'white'}, whiskerprops={'linewidth': 3},
                saturation=0.75)
    plt.ylim(0, 2000)  # 这里假设你想将y轴范围设置为0到200

    # 设置y轴的刻度位置和标签
    plt.yticks([0, 500, 1000, 1500, 2000], ['0', '500', '1000', '1500', '2000'])
    plt.tight_layout()
    plt.savefig(violin_plot_savepath + 'OPTI_statistics.png')
    plt.close()


def heat_map(data_df):
    SHAP_plot_save_path = './SHAP_plot2/'
    heat_map_path = SHAP_plot_save_path + "heatmap/"
    plt.figure(dpi=1000, figsize=(10, 10))
    plt.rcParams['font.size'] = 10
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # data_heat = np.corrcoef(data_df.values, rowvar=0)
    data_heat, p_value = spearmanr(data_df.values, axis=0)
    data_heat = pd.DataFrame(data=data_heat, columns=data_df.columns, index=data_df.columns)
    plt.figure(figsize=(10, 10))
    plt.rcParams['font.family'] = 'Arial'
    colors = ["#581113", "#F6F6F6", "#202050"]
    n_colors = 100
    cmap = LinearSegmentedColormap.from_list("", ["#202050", "#1B5AAB", "#4EA2D2", "#BCD6EC", "#F6F6F6", "#FECEB3",
                                                  "#EF8565", "#B7201E", "#581113"])
    # all features
    ax = sns.heatmap(np.round(data_heat, 2), square=True, annot=True, fmt='.2f', linewidths=.5, cmap=cmap,annot_kws={"size":12},
                     cbar_kws={'fraction': 0.046, 'pad': 0.03,'ticks': np.linspace(-1, 4, 11)}, vmin=-1, vmax=1,
                     xticklabels=["Ca", "Cl", "Si", "Al", "MC time", "MC speed", "L/S", "BW/W", "Ca-additive", "P-additive", "Ca-P additive",
    "Si-Al additive","Leaching pH","Total metal","Initial concentration", "Ionic radius", "Electronegativity", "OPTI"],
                     yticklabels=["Ca", "Cl", "Si", "Al", "MC time", "MC speed", "L/S", "BW/W", "Ca-additive", "P-additive", "Ca-P additive",
    "Si-Al additive","Leaching pH","Total metal","Initial concentration", "Ionic radius", "Electronegativity", "OPTI"])
    plt.xticks(fontsize=16,rotation=45, rotation_mode='anchor',ha='right')
    plt.yticks(fontsize=16)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.savefig(heat_map_path + 'heatmap-all features.png', bbox_inches='tight')