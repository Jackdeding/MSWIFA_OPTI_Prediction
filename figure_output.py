import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from matplotlib.ticker import MultipleLocator, MaxNLocator
from shap import summary_plot,heatmap_plot, dependence_plot
from shap.plots import heatmap, scatter , waterfall
import shap
from sklearn.inspection import PartialDependenceDisplay

SHAP_plot_save_path = './SHAP_plot2/'
feature_SHAP_plot_path = 'All feature SHAP scatter plots/'
dependence_plot_save_path = 'SHAP dependence_plot/'

def get_label_name(feature_name):
    ignore_list = ["Ca", "Cl", "Si", "Al", "MC time", "MC speed", "L/S", "BW/W", "Ca-additive", "P-additive", "Ca-P additive",
    "Si-Al additive","Leaching pH","Total metal","Initial concentration", "Ionic radius", "Electronegativity", "OPTI"]
    if feature_name in ignore_list:
        return feature_name
    else:
        return feature_name

def feature_name_replace(feature_name: str):
    if type(feature_name) is str:
        return feature_name.replace('/', '-')
    else:
        print("\033[32mTypeError: Type of parameter 'feature_name' is str, input value is " + str(type(feature_name))
              + "\033[0m")
def predict_compare_plot(train_target, train_predict, test_target, test_predict):
    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 18
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.family'] = 'Arial'
    colors = sns.color_palette("colorblind")
    test_result = pd.concat([pd.DataFrame(test_target.values), pd.DataFrame(test_predict)], axis=1)
    train_result = pd.concat([pd.DataFrame(train_target.values), pd.DataFrame(train_predict)], axis=1)
    plt.plot([0, 2400], [0, 2400], linestyle='--', alpha=0.85, c='black', linewidth=2)
    plt.scatter(train_result.iloc[:, 0], train_result.iloc[:, 1], alpha=1, c='w',edgecolors=colors[0], s=100, label='Training',facecolors='DE8F05')
    plt.scatter(test_result.iloc[:, 0], test_result.iloc[:, 1], marker='^', alpha=1, c='w', edgecolors=colors[1], s=100, label='Test')
    plt.tick_params(labelsize=16)
    plt.xlabel('Actual OPTI', fontsize='18')
    plt.ylabel('Predicted OPTI', fontsize='18')
    plt.legend(loc=2, fontsize=18, markerscale=1.1, frameon=False)
    plt.grid(False)
    x = MultipleLocator(300)
    y = MultipleLocator(300)
    ax = plt.gca()
    plt.xlim((0, 2400))
    plt.ylim((0, 2400))
    ax.yaxis.set_major_locator(y)
    ax.xaxis.set_major_locator(x)
    # 画横坐标上值为 40、80、160、320 的垂直虚线，在对角线处停止
    for x in [40, 80, 160, 320]:
        plt.plot([x, x], [0, x], linestyle='--', color='gray', alpha=0.5)

    # 画纵坐标上值为 40、80、160、320 的水平虚线，在对角线处停止
    for y in [40, 80, 160, 320]:
        plt.plot([0, y], [y, y], linestyle='--', color='gray', alpha=0.5)

    # 计算并添加R2值
    train_r2 = r2_score(train_result.iloc[:, 0], train_result.iloc[:, 1])
    test_r2 = r2_score(test_result.iloc[:, 0], test_result.iloc[:, 1])
    plt.text(1600, 200, "Train R2: {:.3f}\nTest R2: {:.3f}".format(train_r2, test_r2), fontsize=14)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "predict.png")
    plt.close()

    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 18
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.family'] = 'Arial'
    colors = sns.color_palette("colorblind")
    test_result = pd.concat([pd.DataFrame(test_target.values), pd.DataFrame(test_predict)], axis=1)
    train_result = pd.concat([pd.DataFrame(train_target.values), pd.DataFrame(train_predict)], axis=1)

    plt.plot([0, 400], [0, 400], linestyle='--', alpha=0.85, c='black', linewidth=2)
    plt.scatter(train_result.iloc[:, 0], train_result.iloc[:, 1], alpha=1, c='w',edgecolors=colors[0], s=100, label='Training',facecolors='DE8F05')
    plt.scatter(test_result.iloc[:, 0], test_result.iloc[:, 1], marker='^', alpha=1, c='w', edgecolors=colors[1], s=100, label='Test')
    plt.tick_params(labelsize=16)
    plt.xlabel('Actual OPTI', fontsize='18')
    plt.ylabel('Predicted OPTI', fontsize='18')
    plt.legend(loc=2, fontsize=18, markerscale=1.3, frameon=False)
    plt.grid(False)
    x = MultipleLocator(40)
    y = MultipleLocator(40)
    ax = plt.gca()
    plt.xlim((0, 400))
    plt.ylim((0, 400))
    ax.yaxis.set_major_locator(y)
    ax.xaxis.set_major_locator(x)
    # 画横坐标上值为 40、80、160、320 的垂直虚线，在对角线处停止
    for x in [40, 80, 160, 320]:
        plt.plot([x, x], [0, x], linestyle='--', color='b', alpha=0.5)

    # 画纵坐标上值为 40、80、160、320 的水平虚线，在对角线处停止
    for y in [40, 80, 160, 320]:
        plt.plot([0, y], [y, y], linestyle='--', color='b', alpha=0.5)

    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "predict-400.png")
    plt.close()

def test_predict_plot(test_predict, test_target):
    plt.figure(dpi=600, figsize=(16, 9))
    plt.rcParams['font.size'] = 18
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.family'] = 'Arial'
    sample_number = []
    for number in range(len(test_predict)):
        sample_number.append(number+1)
    colors = sns.color_palette("colorblind")
    plt.plot(sample_number, test_target, label="Actual OPTI", linewidth=3, linestyle='-', marker='o', markersize='10')
    plt.plot(sample_number, test_predict, color=colors[1], label="Predicted OPTI", linewidth=3, linestyle='--', marker='^', markersize='10')
    plt.xlabel("Sample Number", fontsize=30)
    plt.ylabel("OPTI", fontsize=30)
    plt.legend(loc='upper right', fontsize=20, markerscale=2, frameon=False)
    plt.tick_params(labelsize=30)
    plt.grid(False)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim((0, sample_number.append(number+1)))
    plt.ylim((0, 2500))
    x = MultipleLocator(25)
    y = MultipleLocator(250)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y)
    ax.xaxis.set_major_locator(x)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "test_predict.png")
    plt.close()

def relevance_plot(train_shap_values, feature_train, feature_name1, feature_name2):
    feature_values1 = feature_train[str(feature_name1)].values
    feature_values2 = feature_train[str(feature_name2)].values
    feature_name1_index = int(feature_train.columns.get_loc(str(feature_name1)))
    feature_name2_index = int(feature_train.columns.get_loc(str(feature_name2)))
    shap_values_sum = train_shap_values[:, feature_name1_index] + train_shap_values[:, feature_name2_index]
    bottom = shap_values_sum.min() - 1
    top = shap_values_sum.max() + 1
    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 10
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    ax1 = plt.axes(projection='3d')
    ax1.set_zlim(bottom, top)
    im = ax1.scatter3D(feature_values1, feature_values2, shap_values_sum, c=shap_values_sum, cmap='jet')
    ax1.scatter3D(feature_values1, feature_values2, bottom - 1)
    ax1.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    ax1.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    ax1.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    plt.grid(True)
    plt.grid(alpha=0.2)
    for number in range(len(shap_values_sum)):
        xs = [feature_values1[number], feature_values1[number]]
        ys = [feature_values2[number], feature_values2[number]]
        zs = [shap_values_sum[number], bottom - 1]
        plt.plot(xs, ys, zs, c='grey', linestyle='--', alpha=0.1, linewidth=0.8)
    plt.tick_params(labelsize=13, pad=0.1)
    plt.xlabel(str(feature_name1), fontsize=15)
    plt.ylabel(str(feature_name2), fontsize=15)
    plt.colorbar(im, fraction=0.1, shrink=0.6, pad=0.1)
    ax1.view_init(elev=20)
    plt.savefig(SHAP_plot_save_path + str(feature_name1) + "_" + str(feature_name2) + ".png")

def material_relevance_plot(feature_train_1, feature_train_2, train_shap_values_1, train_shap_values_2, feature_name1, feature_name2):
    shap_values = train_shap_values_1 + train_shap_values_2
    bottom = int(shap_values.min()) - 1
    top = int(shap_values.max()) + 1
    c = shap_values
    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 13
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    ax1 = plt.axes(projection='3d')
    ax1.set_zlim(bottom, top)
    im = ax1.scatter3D(feature_train_1, feature_train_2, shap_values, c=c, cmap='jet')
    ax1.scatter3D(feature_train_1, feature_train_2, -25)
    plt.grid(True)
    plt.grid(alpha=0.2)
    for number in range(len(shap_values)):
        xs = [feature_train_1[number], feature_train_1[number]]
        ys = [feature_train_2[number], feature_train_2[number]]
        zs = [shap_values[number], bottom]
        plt.plot(xs, ys, zs, c='grey', linestyle='--', alpha=0.1, linewidth=0.8)
    plt.tick_params(labelsize=13, pad=0.1)
    plt.xlabel(feature_name1, fontsize=15)
    plt.ylabel(feature_name2, fontsize=15)
    plt.colorbar(im, fraction=0.1, shrink=0.6, pad=0.1)
    ax1.view_init(elev=20)
    plt.savefig(SHAP_plot_save_path + feature_name1 + "_" + feature_name2 + ".png")

def calculate_material(feature_name_1, feature_name_2, train_shap_values, feature_train):
    feature1_num = int(feature_train.columns.get_loc(str(feature_name_1)))
    feature2_num = int(feature_train.columns.get_loc(str(feature_name_2)))
    feature_sum_value = feature_train[str(feature_name_1)].values + feature_train[str(feature_name_2)].values
    shap_sum_value = train_shap_values[:, feature1_num] + train_shap_values[:, feature2_num]
    return feature_sum_value, shap_sum_value

def get_material_shap_value(feature_name, train_shap_value, feature_train):
    feature_num = int(feature_train.columns.get_loc(str(feature_name)))
    feature_value = feature_train[str(feature_name)].values
    feature_shap_value = train_shap_value[:, feature_num]
    return feature_value, feature_shap_value

def SHAP_plot(predict_model, raw_feature_train_summary, raw_feature_train):
    explainer = shap.KernelExplainer(predict_model, raw_feature_train_summary)
    shap_values = explainer.shap_values(raw_feature_train)
    #shap_values = explainer.shap_values(raw_feature_train)
    # feature_SHAP(9, feature_train['Fe2O3'], shap_values)
    # All features SHAP scatter plots
    feature_name_list = ["Ca", "Cl", "Si", "Al", "MC time", "MC speed", "L/S", "BW/W", "Ca-additive", "P-additive", "Ca-P additive",
    "Si-Al additive","Leaching pH","Total metal","Initial concentration", "Ionic radius", "Electronegativity"]
    for feature_name in feature_name_list:
        plt.figure(dpi=600, figsize=(10, 10))
        plt.rcParams['font.family'] = 'Arial'
        feature_value, SHAP_value = get_material_shap_value(feature_name, shap_values, raw_feature_train)
        data = {
            'feature_value': feature_value,
            'SHAP_value': SHAP_value
                }
        df = pd.DataFrame(data)
        df.to_excel(SHAP_plot_save_path + feature_SHAP_plot_path + feature_name_replace(feature_name) + " scatter.xlsx")
        plt.scatter(feature_value, SHAP_value, s=150, cmap='plasma',alpha=0.5)
        plt.tick_params(labelsize=20, pad=0.1)
        plt.xlabel(get_label_name(feature_name=feature_name) + " Value", fontsize=20)
        plt.ylabel("SHAP Value of " + get_label_name(feature_name=feature_name), fontsize=20)
        plt.tight_layout()
        plt.savefig(SHAP_plot_save_path + feature_SHAP_plot_path + feature_name_replace(feature_name) + " scatter.png")
        plt.close()

    # Dot summary plot
    plt.figure(dpi=600, figsize=(8, 8))
    plt.rcParams['font.size'] = 25
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams["axes.labelweight"] = "bold"
    plt.tick_params(labelsize=25)
    summary_plot(shap_values, raw_feature_train, plot_type='dot', show=False)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP Feature Importance.png")
    plt.close()

    # Bar summary plot
    plt.figure(dpi=600, figsize=(8, 8))
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    summary_plot(shap_values, raw_feature_train, plot_type='bar', show=False)
    fig = plt.gcf()
    ax = plt.gca()
    bars = ax.patches
    plt.rcParams["axes.labelweight"] = "bold"
    plt.tick_params(labelsize=16)
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                va='center', ha='left', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP Feature Importance1.png")
    plt.close()

def feature_SHAP(feature_number, feature_name, feature_train, SHAP_values):
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt_x = feature_train
    feature_SHAP_value = SHAP_values[:, feature_number]
    plt.scatter(plt_x, feature_SHAP_value, s=10,dot_size=50,color=SHAP_values[:, feature_number])
    plt.xlabel(feature_name + " Value")
    plt.ylabel("SHAP Value of " + feature_name)
    plt.savefig(SHAP_plot_save_path + feature_name_replace(feature_name=feature_name) + ".png")
    plt.close()

def other_SHAP_plot(predict_model, raw_feature_train_summary, raw_feature_train,raw_train_target,raw_test_feature,raw_test_target):
    fig = plt.figure(dpi=600, figsize=(10, 10))
    # expleiner = shap.Explainer(predict_model, feature_train)
    # shap_values = expleiner(feature_train)
    feature_name_list = ["Ca", "Cl", "Si", "Al", "MC time", "MC speed", "L/S", "BW/W", "Ca-additive", "P-additive", "Ca-P additive",
    "Si-Al additive","Leaching pH","Total metal","Initial concentration", "Ionic radius", "Electronegativity"]
    explainer = shap.KernelExplainer(predict_model, raw_feature_train_summary)
    shap_values = explainer(raw_feature_train)
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    heatmap(shap_values, show=False, max_display=17)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_heatmap.png")
    plt.close()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.sans-serif'] = ['Arial']
    waterfall(shap_values[100], show=False, max_display=20)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_partial_waterfall_plot.png")
    plt.close()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.sans-serif'] = ['Arial']
    shap.plots.bar(shap_values[100], show=False, max_display=17)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_partial_FI_plot.png")
    plt.close()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.sans-serif'] = ['Arial']
    shap.force_plot(explainer.expected_value, shap_values.values, raw_feature_train)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "ALL SHAP_partial_FI_plot.png")
    plt.close()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.size'] = 15
    plt.rcParams['font.sans-serif'] = ['Arial']
    all_target = np.concatenate((raw_train_target, raw_test_target), 0)
    all_feature = np.concatenate((raw_feature_train, raw_test_feature), 0)
    clustering = shap.utils.hclust(all_feature, all_target)
    shap.plots.bar(shap_values,clustering=clustering, clustering_cutoff=0.5,show=False)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_feature_clustering_plot.png")

    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "MC time"], color=shap_values[:, "Initial concentration"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_1.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "MC time"], color=shap_values[:, "Ca-additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_2.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "MC time"], color=shap_values[:, "P-additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_3.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "MC time"], color=shap_values[:, "Ca-P additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_4.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "MC time"], color=shap_values[:, "Si-Al additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_5.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "MC time"], color=shap_values[:, "Leaching pH"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_6.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "Initial concentration"], color=shap_values[:, "MC time"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_61.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "Initial concentration"], color=shap_values[:, "Ca-additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_7.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "Initial concentration"], color=shap_values[:, "P-additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_8.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "Initial concentration"], color=shap_values[:, "Ca-P additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_9.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "Initial concentration"], color=shap_values[:, "Si-Al additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_10.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "Leaching pH"], color=shap_values[:, "MC time"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_111.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "Leaching pH"], color=shap_values[:, "Ca-additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_11.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "Leaching pH"], color=shap_values[:, "P-additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_12.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "Leaching pH"], color=shap_values[:, "Ca-P additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_13.png")
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.plots.scatter(shap_values[:, "Leaching pH"], color=shap_values[:, "Si-Al additive"], show=False,dot_size=50)
    # plt.savefig(SHAP_plot_save_path + "dependence_14.png")
    plt.close()

    explainer = shap.KernelExplainer(predict_model, raw_feature_train_summary)
    shap_values = explainer.shap_values(raw_feature_train)


    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.dependence_plot('Ca', shap_values, raw_feature_train,interaction_index='Initial concentration', show=False,dot_size=50)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig('./SHAP_plot2/SHAP dependence_plot/Ca_dependence.png')
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.dependence_plot('Si', shap_values, raw_feature_train, show=False,dot_size=50)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig('./SHAP_plot2/SHAP dependence_plot/Si_dependence.png')
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.dependence_plot('Al', shap_values, raw_feature_train, show=False,dot_size=50)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig('./SHAP_plot2/SHAP dependence_plot/Al_dependence.png')
    # plt.close()
    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Cl', shap_values, raw_feature_train,interaction_index='Initial concentration', show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Cl_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('MC time', shap_values, raw_feature_train, interaction_index='Initial concentration',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/MC time_dependence.png')
    plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.dependence_plot('MC speed', shap_values, raw_feature_train, show=False,dot_size=50)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig('./SHAP_plot2/SHAP dependence_plot/MC speed_dependence.png')
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.dependence_plot('L/S', shap_values, raw_feature_train, show=False,dot_size=50)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig('./SHAP_plot2/SHAP dependence_plot/L-S_dependence.png')
    # plt.close()
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.dependence_plot('BW/W', shap_values, raw_feature_train, show=False,dot_size=50)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig('./SHAP_plot2/SHAP dependence_plot/BW-W_dependence.png')
    # plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Ionic radius', shap_values, raw_feature_train, interaction_index='Initial concentration',
                         show=False, dot_size=150, alpha=0.5)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Ca-additive_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Ca-additive', shap_values, raw_feature_train, interaction_index='Initial concentration',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Ca-additive_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Ca-P additive', shap_values, raw_feature_train, interaction_index='Initial concentration',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Ca-P additive_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('P-additive', shap_values, raw_feature_train, interaction_index='Initial concentration',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/P-additive_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Si-Al additive', shap_values, raw_feature_train, interaction_index='Initial concentration',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Si-Al additive_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Leaching pH', shap_values, raw_feature_train, interaction_index='Initial concentration',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Leaching pH_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Initial concentration', shap_values, raw_feature_train,interaction_index='MC time',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    axes = plt.gcf().axes
    # 找到颜色条对应的坐标轴（通常是最后一个坐标轴）
    cbar_ax = axes[-1]
    # 按照 4 等分设置刻度和标签
    num_divisions = 6
    tick_values = np.linspace(raw_feature_train['MC time'].min(), raw_feature_train['MC time'].max(),
                              num_divisions + 1)  # 计算 5 等分点
    cbar_ax.set_yticks(tick_values)
    cbar_ax.set_yticklabels([f'{tick:.0f}' for tick in tick_values])
    axes = plt.gcf().axes
    # 找到颜色条对应的坐标轴（通常是最后一个坐标轴）
    cbar_ax = axes[-1]
    # 按照 4 等分设置刻度和标签
    num_divisions = 6
    tick_values = np.linspace(raw_feature_train['MC time'].min(), raw_feature_train['MC time'].max(),
                              num_divisions + 1)  # 计算 5 等分点
    cbar_ax.set_yticks(tick_values)
    cbar_ax.set_yticklabels([f'{tick:.0f}' for tick in tick_values])
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Initial concentration_dependence.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Ca-additive', shap_values, raw_feature_train, interaction_index='MC time',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    axes = plt.gcf().axes
    # 找到颜色条对应的坐标轴（通常是最后一个坐标轴）
    cbar_ax = axes[-1]
    # 按照 4 等分设置刻度和标签
    num_divisions = 6
    tick_values = np.linspace(raw_feature_train['MC time'].min(), raw_feature_train['MC time'].max(), num_divisions + 1)  # 计算 5 等分点
    cbar_ax.set_yticks(tick_values)
    cbar_ax.set_yticklabels([f'{tick:.0f}' for tick in tick_values])
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Ca-additive_dependence2.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Ca-P additive', shap_values, raw_feature_train, interaction_index='MC time',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    axes = plt.gcf().axes
    # 找到颜色条对应的坐标轴（通常是最后一个坐标轴）
    cbar_ax = axes[-1]
    # 按照 4 等分设置刻度和标签
    num_divisions = 6
    tick_values = np.linspace(raw_feature_train['MC time'].min(), raw_feature_train['MC time'].max(),
                              num_divisions + 1)  # 计算 5 等分点
    cbar_ax.set_yticks(tick_values)
    cbar_ax.set_yticklabels([f'{tick:.0f}' for tick in tick_values])
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Ca-P additive_dependence2.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('P-additive', shap_values, raw_feature_train, interaction_index='MC time',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    axes = plt.gcf().axes
    # 找到颜色条对应的坐标轴（通常是最后一个坐标轴）
    cbar_ax = axes[-1]
    # 按照 4 等分设置刻度和标签
    num_divisions = 6
    tick_values = np.linspace(raw_feature_train['MC time'].min(), raw_feature_train['MC time'].max(), num_divisions + 1)  # 计算 5 等分点
    cbar_ax.set_yticks(tick_values)
    cbar_ax.set_yticklabels([f'{tick:.0f}' for tick in tick_values])
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/P-additive_dependence2.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Si-Al additive', shap_values, raw_feature_train, interaction_index='MC time',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    axes = plt.gcf().axes
    # 找到颜色条对应的坐标轴（通常是最后一个坐标轴）
    cbar_ax = axes[-1]
    # 按照 4 等分设置刻度和标签
    num_divisions = 6
    tick_values = np.linspace(raw_feature_train['MC time'].min(), raw_feature_train['MC time'].max(), num_divisions + 1)  # 计算 5 等分点
    cbar_ax.set_yticks(tick_values)
    cbar_ax.set_yticklabels([f'{tick:.0f}' for tick in tick_values])
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Si-Al additive_dependence2.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.dependence_plot('Leaching pH', shap_values, raw_feature_train, interaction_index='MC time',show=False,dot_size=150,alpha=0.5)
    plt.tick_params(labelsize=15)
    axes = plt.gcf().axes
    # 找到颜色条对应的坐标轴（通常是最后一个坐标轴）
    cbar_ax = axes[-1]
    # 按照 4 等分设置刻度和标签
    num_divisions = 6
    tick_values = np.linspace(raw_feature_train['MC time'].min(), raw_feature_train['MC time'].max(), num_divisions + 1)  # 计算 5 等分点
    cbar_ax.set_yticks(tick_values)
    cbar_ax.set_yticklabels([f'{tick:.0f}' for tick in tick_values])
    plt.tight_layout()
    plt.savefig('./SHAP_plot2/SHAP dependence_plot/Leaching pH_dependence2.png')
    plt.close()


    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.dependence_plot('Total metal', shap_values, raw_feature_train, show=False,dot_size=50)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig('./SHAP_plot2/SHAP dependence_plot/Total metal_dependence.png')
    # plt.close()
    #
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.dependence_plot('Ionic radius', shap_values, raw_feature_train, show=False,dot_size=50)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig('./SHAP_plot2/SHAP dependence_plot/Ionic radius_dependence.png')
    # plt.close()
    #
    # fig = plt.figure(dpi=600, figsize=(6, 6))
    # shap.dependence_plot('Electronegativity', shap_values, raw_feature_train, show=False,dot_size=50)
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig('./SHAP_plot2/SHAP dependence_plot/Electronegativity_dependence.png')
    # plt.close()

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    shap.plots.force(explainer.expected_value, shap_values[100], features=feature_name_list, show=False,matplotlib=True)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_partial_force_plot.png")
    plt.close()

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    shap.decision_plot(explainer.expected_value, shap_values[100], feature_names = feature_name_list,show=False)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_partial_decision_plot.png")
    plt.close()

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    shap.decision_plot(explainer.expected_value, shap_values, feature_names = feature_name_list,show=False)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "SHAP_all_decision_plot.png")
    plt.close()

