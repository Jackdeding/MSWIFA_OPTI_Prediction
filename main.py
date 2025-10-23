import pandas as pd
import shap
from data_processing import scaler_data_processing
from model import DT_predict, MLP_predict, RF_predict, SVR_predict, GB_predict, KNN_predict, XGB_predict, LGBM_predict
from figure_output import SHAP_plot, predict_compare_plot, other_SHAP_plot, test_predict_plot,relevance_plot
from data_statistics import violin_plot, heat_map
import os
import pickle
import joblib
import warnings

warnings.filterwarnings("ignore")
SHAP_plot_save_path = './SHAP_plot2/'
feature_SHAP_plot_path = 'All feature SHAP scatter plots/'
heat_map_path = "heatmap/"
dependence_plot_save_path = 'SHAP dependence_plot/'
violin_plot_savepath = 'dataset_statistics_violin_plot/'
dataset_file = 'D:\machine-learning\MSWIFA_HM_immobilization_prediction_19Feature_0324\DATA0327.xlsx'

def result_plot(model_predict, raw_feature_train_summary, train_target, train_data_predict, test_target, test_data_predict,raw_train_feature, raw_train_target, raw_test_feature, raw_test_target):
    test_predict_plot(test_data_predict, test_target)
    predict_compare_plot(train_target, train_data_predict, test_target, test_data_predict)
    shap.initjs()
    SHAP_plot(model_predict, raw_feature_train_summary, raw_train_feature)
    other_SHAP_plot(model_predict, raw_feature_train_summary, raw_train_feature,raw_train_target,raw_test_feature,raw_test_target)
    # explainer = shap.KernelExplainer(model_predict, raw_train_feature)
    # shap_values = explainer.shap_values(raw_train_feature)
    # relevance_plot(shap_values.values, raw_train_feature,"MC time", "Ca-P additive")
    # relevance_plot(shap_values.values, raw_train_feature, "MC time", "Initial concentration")
    # relevance_plot(train_shap_values, feature_train, feature_name1, feature_name2)
    # material_relevance_plot(feature_train_1, feature_train_2, train_shap_values_1, train_shap_values_2, feature_name1,
    #                         feature_name2)

def run_model():
    # 获取归一化以后的训练集特征值；训练集目标值；归一化后的测试集特征值；测试集目标值；聚类算法的返回值；原始训练集的特征值，原始测试集的特征值
    scalar_training_feature_value, training_target_value,scalar_test_feature_value, test_target_value, raw_feature_train_summary, raw_training_feature_value,raw_test_feature_value,data= scaler_data_processing(dataset_file)
    data_df= pd.DataFrame(data)
    # violin_plot(data_df)
    # input_feature = pd.concat([scalar_training_feature_value, scalar_test_feature_value], ignore_index=True)
    # heat_map(input_feature)
    # heat_map(data_df)

    #model_predict, train_data_predict, test_data_predict, DT_model = DT_predict(scalar_training_feature_value,training_target_value,scalar_test_feature_value,test_target_value)
    #model_predict, train_data_predict, test_data_predict, MLP_model = MLP_predict(scalar_training_feature_value,training_target_value,scalar_test_feature_value,test_target_value)
    #model_predict, train_data_predict, test_data_predict, RF_model = RF_predict(scalar_training_feature_value,training_target_value,scalar_test_feature_value,test_target_value)
    # model_predict, train_data_predict, test_data_predict, KNN_model = KNN_predict(scalar_training_feature_value,training_target_value,scalar_test_feature_value,test_target_value)
    #model_predict, train_data_predict, test_data_predict,SVR_modle = SVR_predict(scalar_training_feature_value,training_target_value,scalar_test_feature_value,test_target_value)
    # model_predict, train_data_predict, test_data_predict, GB_model = GB_predict(scalar_training_feature_value,training_target_value,scalar_test_feature_value,test_target_value)
    model_predict, train_data_predict, test_data_predict, XGB_model = XGB_predict(scalar_training_feature_value,training_target_value,scalar_test_feature_value,test_target_value)
    # model_predict, train_data_predict, test_data_predict, LGBM_model = LGBM_predict(scalar_training_feature_value,training_target_value,scalar_test_feature_value,test_target_value)
    joblib.dump(XGB_model, 'XGB_model.joblib')
    # joblib.dump(GB_model, 'Predict_model.joblib' )
    # joblib.dump(scaler_data_processing(), 'scaler_data_processing.joblib')
    # pickle.dump(model_predict, open("Predict.pickle.dat", "wb"))
    #joblib.dump(trained_model, 'trained_model1.pkl' )
    result_plot(model_predict, raw_feature_train_summary, training_target_value,
                train_data_predict, test_target_value,test_data_predict,
                raw_training_feature_value,training_target_value,raw_test_feature_value,test_target_value)

# Make a directory to save result picture
def make_directory():
    try:
        if not os.path.exists(SHAP_plot_save_path):
            os.mkdir(SHAP_plot_save_path)
        if not os.path.exists(SHAP_plot_save_path + feature_SHAP_plot_path):
            os.mkdir(SHAP_plot_save_path + feature_SHAP_plot_path)
        if not os.path.exists(SHAP_plot_save_path + heat_map_path):
            os.mkdir(SHAP_plot_save_path + heat_map_path)
        if not os.path.exists(SHAP_plot_save_path + dependence_plot_save_path):
            os.mkdir(SHAP_plot_save_path + dependence_plot_save_path)
        if os.path.exists(SHAP_plot_save_path) and os.path.exists(SHAP_plot_save_path + feature_SHAP_plot_path)\
                and os.path.exists(SHAP_plot_save_path + heat_map_path) and os.path.exists(SHAP_plot_save_path + dependence_plot_save_path):
            print("The SHAP plot directory already exists.")
        else:
            print("Successfully created folder, path is: " + SHAP_plot_save_path)
    except Exception as mkdir_error:
        print("Make Directory Error: " + str(mkdir_error))

def Initialization():
    print("Initialize...")
    make_directory()


if __name__ == "__main__":
    Initialization()
    run_model()

