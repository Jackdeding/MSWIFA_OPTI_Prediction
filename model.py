import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import sort
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectFromModel

def MAPE(y, y_pre):
    y = np.array(y)
    y_pre = np.array(y_pre)
    return np.mean(np.abs((y_pre - y ) / y )) * 100
def SMAPE(y, y_pre):
    y = np.array(y)
    y_pre = np.array(y_pre)
    return 2.0 * np.mean(np.abs(y_pre - y) / (np.abs(y_pre) + np.abs(y))) * 100

def DT_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=False)
    param_grid = {
        'ccp_alpha': [0.05],
        'splitter': ['best'],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'random_state': [5],
        'max_features': [20],
    }
    #2025.3.23 add test
    # model = DecisionTreeRegressor()
    # grid_search = GridSearchCV(model, param_grid, cv=kf)
    # grid_search.fit(feature_train, target_train)
    # best_model = grid_search.best_estimator_
    # # 在训练集上进行预测
    # train_pred = best_model.predict(feature_train)
    # # 计算训练集的评估指标
    # train_rmse = mean_squared_error(target_train, train_pred, squared=False)
    # train_mae = mean_absolute_error(target_train, train_pred)
    # train_r2 = r2_score(target_train, train_pred)
    # train_mape = mean_absolute_percentage_error(target_train, train_pred)
    #
    # print(f"Train data RMSE:{train_rmse}")
    # print(f"Train data MAE:{train_mae}")
    # print(f"Train data R2:{train_r2}")
    # print(f"Train data MAPE:{train_mape}")
    #
    # # 在测试集上进行预测
    # test_pred = best_model.predict(feature_test)
    # # 计算测试集的评估指标
    # test_rmse = mean_squared_error(target_test, test_pred, squared=False)
    # test_mae = mean_absolute_error(target_test, test_pred)
    # test_r2 = r2_score(target_test, test_pred)
    # test_mape = mean_absolute_percentage_error(target_test, test_pred)
    #
    # print(f"Test data RMSE:{test_rmse}")
    # print(f"Test data MAE:{test_mae}")
    # print(f"Test data R2:{test_r2}")
    # print(f"Test data MAPE:{test_mape}")
    #
    # return best_model

    DT_model = DecisionTreeRegressor()
    grid_search = GridSearchCV(DT_model, param_grid, cv=kf, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    DT_model = grid_search.best_estimator_
    DT_model.fit(feature_train, target_train)
    train_predict = DT_model.predict(feature_train)
    test_predict = DT_model.predict(feature_test)
    cross_val_RMSE = -1 * cross_val_score(DT_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(DT_model, feature_train, target_train, cv=kf,
                                          scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(DT_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = DT_model.predict(feature_train)
    test_predict = DT_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = DT_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = DT_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return DT_model.predict, train_predict, test_predict, DT_model

def MLP_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle= False)
    param_grid = {"solver": ["lbfgs"],
                  "max_iter": [4000],
                  "activation": ["relu"],
                  "alpha": [0.01],
                  "hidden_layer_sizes": [(20,8,3)],
                  "learning_rate": ["constant"],
                  "random_state": [3],
                  # "warm_start": [True]
                  }
    MLP_model = MLPRegressor()
    # r2 = make_scorer(r2_score, greater_is_better = True)
    grid_search = GridSearchCV(MLP_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    MLP_model = grid_search.best_estimator_
    MLP_model.fit(feature_train, target_train)
    R2_train_score = MLP_model.score(feature_train, target_train)
    print("MLP R2 Train Scores: " + str(R2_train_score))
    # R2_train_score = r2_score(target_train, MLP_model.predict(feature_train))
    cross_val_RMSE = -1 * cross_val_score(MLP_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(MLP_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(MLP_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = MLP_model.predict(feature_train)
    test_predict = MLP_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = MLP_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = MLP_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return MLP_model.predict, train_predict, test_predict, MLP_model

def RF_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    param_grid = {"n_estimators": [30,300,500],
                  "max_depth": [5,9],
                  "min_samples_split": [4],
                  "min_samples_leaf": [2],
                  "max_features": [20]
                  }
    RF_model = RandomForestRegressor()
    grid_search = GridSearchCV(RF_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    RF_model = grid_search.best_estimator_
    RF_model.fit(feature_train, target_train)
    R2_train_score = RF_model.score(feature_train, target_train)
    print("RF R2 Train Scores: " + str(R2_train_score))
    cross_val_RMSE = -1 * cross_val_score(RF_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(RF_model, feature_train, target_train, cv=kf,
                                      scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(RF_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = RF_model.predict(feature_train)
    test_predict = RF_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = RF_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = RF_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return RF_model.predict, train_predict, test_predict, RF_model

def KNN_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    param_grid = {'algorithm': ['kd_tree'],
                    'leaf_size': [10],
                    'weights': ['uniform'],
                    'n_neighbors': [4],
                 }
    KNN_model = KNeighborsRegressor()
    grid_search = GridSearchCV(KNN_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    KNN_model = grid_search.best_estimator_
    KNN_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(KNN_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(KNN_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(KNN_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = KNN_model.predict(feature_train)
    test_predict = KNN_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = KNN_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = KNN_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return KNN_model.predict, train_predict, test_predict, KNN_model

def SVR_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=5)
    param_grid = {
        'kernel': ['rbf'],
        'C': [10000],
        'gamma': [1],
        'epsilon': [0.1],
        'max_iter':[10000],
    }
    SVR_model = SVR()
    grid_search = GridSearchCV(SVR_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    SVR_model = grid_search.best_estimator_
    SVR_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(SVR_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(SVR_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(SVR_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = SVR_model.predict(feature_train)
    test_predict = SVR_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = SVR_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = SVR_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return SVR_model.predict, train_predict, test_predict, SVR_model

def GB_predict(feature_train, target_train, feature_test, target_test):
    target_train = np.ravel(target_train)
    target_test = np.ravel(target_test)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    param_grid = {
        "n_estimators": [500,800,1000],
        "learning_rate": [0.1],
        "random_state": [47],
        "alpha": [0.01],
        "max_depth":[5],
        "max_features":[None],
        "min_samples_split":[2],
        "min_samples_leaf":[3],
        "subsample":[1],
        # "colsample_bylevel":[0.7,0.8,0.9],
        # "early_stopping_rounds":[5,10],
    }
    GB_model = GradientBoostingRegressor()
    grid_search = GridSearchCV(GB_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    GB_model = grid_search.best_estimator_
    GB_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(GB_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(GB_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(GB_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = GB_model.predict(feature_train)
    test_predict = GB_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = GB_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = GB_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))

    # GB_model = GradientBoostingRegressor()
    # np.random.seed(42)
    # rfecv = RFECV(GB_model, step=1, cv=kf, scoring='r2', importance_getter='feature_importances_' )
    # rfecv.fit(feature_train, target_train)

    # print("Ranking %s" % rfecv.ranking_)
    # print( rfecv.n_features_)
    # print( rfecv.estimator_.feature_importances_)
    # gb_score = rfecv.score(feature_train, target_train)
    # print(gb_score)
    # gb_score1 = rfecv.score(feature_test, target_test)
    # print(gb_score1)
    # print("Grid Scores %s" % rfecv.cv_results_['mean_test_score'])
    # plt.figure()
    # plt.title('RFECV feature ranking change curve')
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score")
    # plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
    # plt.grid(True)
    # plt.show()

    # GB_model = GradientBoostingRegressor()
    # trained_model = GB_model.fit(feature_train, target_train)
    # target_test_pred = GB_model.predict(feature_test)
    # predictions = [round(value) for value in target_test_pred]
    # r2 = r2_score(target_test, predictions)
    # print(GB_model.feature_importances_)
    # plt.bar(range(len(GB_model.feature_importances_)), GB_model.feature_importances_)
    # plt.show()
    # print("R2:"  + str(r2))
    # thresholds = sort(GB_model.feature_importances_)
    # for thresh in thresholds:
    #     # select features using threshold
    #     selection = SelectFromModel(GB_model, threshold=thresh, prefit=True)
    #     select_feature_train = selection.transform(feature_train)
    #
    #     # train model
    #     selection_model = GradientBoostingRegressor()
    #     selection_model.fit(select_feature_train, target_train)
    #
    #     # eval model
    #     select_feature_test = selection.transform(feature_test)
    #     target_test_pred = selection_model.predict(select_feature_test)
    #     predictions = [round(value) for value in target_test_pred]
    #     r2 = r2_score(target_test, predictions)
    #     print("Thresh=%.3f, n=%d, R2: %.2f%%" % (thresh, select_feature_train.shape[1], r2))
    # GB_model = GradientBoostingRegressor()
    # GB_model.fit(feature_train, target_train)
    # importances = GB_model.feature_importances_
    # feature_importances_df = pd.DataFrame({'feature': feature_train.columns, 'importance': importances})
    # feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
    # target_test_pred = GB_model.predict(feature_test)
    # r2 = r2_score(target_test, target_test_pred)
    # print(feature_importances_df)
    # print(r2)
    feature_importance = GB_model.feature_importances_
    feature_name_list = ["Ca", "Cl", "Si", "Al", "MC time", "MC speed", "L/S", "BW/W", "Ca-additive", "P-additive", "Ca-P additive",
    "Si-Al additive","leaching pH","Total metal","Initial concentration", "Ionic radius", "Electronegativity"]
    sorted_indices = np.argsort(feature_importance)
    sorted_feature_importance = np.array(feature_importance)[sorted_indices]
    sorted_feature_name_list = np.array(feature_name_list)[sorted_indices]
    y_pos = np.arange(len(feature_name_list))
    plt.figure(dpi=600, figsize=(19, 19))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    bars = plt.barh(y_pos, sorted_feature_importance, align='center', alpha=0.5)
    plt.xticks(fontsize = 35)
    plt.yticks(y_pos, sorted_feature_name_list, fontsize=35)
    plt.xlabel('Feature importance score', fontsize=35)
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center', ha='left', fontsize=30,
                 fontweight='bold')
    plt.tight_layout()
    SHAP_plot_save_path = './SHAP_plot2/'
    plt.savefig(SHAP_plot_save_path + "GB Feature Importance.png")
    return GB_model.predict, train_predict, test_predict, GB_model

def XGB_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    param_grid = {
        "n_estimators": [3000],
        "learning_rate": [0.01],
         "random_state": [47],
        "alpha": [0.1],
        "max_depth": [6],
        "max_features": [20],
        "max_delta_step":[0],
        "min_child_weight": [1],
        "subsample": [0.8],
        "gamma":[0.1],
    }
    XGB_model = XGBRegressor()
    grid_search = GridSearchCV(XGB_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    XGB_model = grid_search.best_estimator_
    XGB_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(XGB_model, feature_train, target_train, cv=kf,
                                          scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    cross_val_MAE = -1 * cross_val_score(XGB_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(XGB_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = XGB_model.predict(feature_train)
    test_predict = XGB_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = XGB_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = XGB_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))

    feature_importance = XGB_model.feature_importances_
    feature_name_list = ["Ca", "Cl", "Si", "Al", "MC time", "MC speed", "L/S", "BW/W", "Ca-additive", "P-additive", "Ca-P additive",
    "Si-Al additive","Leaching pH","Total metal","Initial concentration", "Ionic radius", "Electronegativity"]
    sorted_indices = np.argsort(feature_importance)
    sorted_feature_importance = np.array(feature_importance)[sorted_indices]
    sorted_feature_name_list = np.array(feature_name_list)[sorted_indices]
    y_pos = np.arange(len(feature_name_list))
    plt.figure(dpi=600, figsize=(19, 19))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    bars = plt.barh(y_pos, sorted_feature_importance, align='center', alpha=0.5)
    plt.xticks(fontsize = 35)
    plt.yticks(y_pos, sorted_feature_name_list, fontsize=35)
    plt.xlabel('Feature importance score', fontsize=35)
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center', ha='left', fontsize=30,
                 fontweight='bold')
    plt.tight_layout()
    SHAP_plot_save_path = './SHAP_plot2/'
    plt.savefig(SHAP_plot_save_path + "XGB Feature Importance.png")

    # return XGB_model.predict, train_predict, test_predict, XGB_model

    return lambda x: XGB_model.predict(x) , train_predict, test_predict, XGB_model

def LGBM_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    param_grid = {
        "num_leaves": [9],
        "n_estimators": [300],
        "learning_rate": [0.1],
        "max_depth": [7],
        "min_child_weight": [1],
        "min child sample": [1],
        "reg_alpha": [0.1],
        "subsample": [0.8],
        "verbosity": [-1],
    }
    LGBM_model = LGBMRegressor()
    grid_search = GridSearchCV(LGBM_model, param_grid, cv=10, scoring='r2')
    grid_search.fit(feature_train, target_train)
    print(grid_search.best_params_)
    LGBM_model = grid_search.best_estimator_
    LGBM_model.fit(feature_train, target_train)
    cross_val_RMSE = -1 * cross_val_score(LGBM_model, feature_train, target_train, cv=kf,
                                        scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()

    cross_val_MAE = -1 * cross_val_score(LGBM_model, feature_train, target_train, cv=kf,
                                         scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    cross_val_R2 = cross_val_score(LGBM_model, feature_train, target_train, cv=kf, scoring='r2')
    Average_R2_score = cross_val_R2.mean()
    train_predict = LGBM_model.predict(feature_train)
    test_predict = LGBM_model.predict(feature_test)
    train_RMSE = mean_squared_error(target_train, train_predict) ** 0.5
    print("Train data RMSE:" + str(train_RMSE))
    train_MAE = mean_absolute_error(target_train, train_predict)
    print("Train data MAE:" + str(train_MAE))
    train_R2 = LGBM_model.score(feature_train, target_train)
    print("Train data R2:" + str(train_R2))
    train_MAPE = MAPE(target_train, train_predict)
    print("Train data MAPE:" + str(train_MAPE))
    train_SMAPE = SMAPE(target_train, train_predict)
    print("Train data SMAPE:" + str(train_SMAPE))
    train_EVS = explained_variance_score(target_train, train_predict)
    print("Train data EVS:" + str(train_EVS))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_R2 = LGBM_model.score(feature_test, target_test)
    print("Test data R2:" + str(test_R2))
    test_MAPE = MAPE(target_test, test_predict)
    print("Test data MAPE:" + str(test_MAPE))
    test_SMAPE = SMAPE(target_test, test_predict)
    print("Test data SMAPE:" + str(test_SMAPE))
    test_EVS = explained_variance_score(target_test, test_predict)
    print("Test data EVS:" + str(test_EVS))
    return lambda x: LGBM_model.predict(x), train_predict, test_predict, LGBM_model