import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import warnings


model = load('D:\machine-learning\MSWIFA_HM_immobilization_prediction_19Feature_0324\ding\Predict_model.joblib')
# model = load('D:\machine-learning\MSWIFA_HM_immobilization_prediction_19Feature_0324\ding\GB_model.joblib')
# model = load('D:\machine-learning\MSWIFA_HM_immobilization_prediction_19Feature_0324\ding\XGB_model.joblib')
scaler = load('D:\machine-learning\MSWIFA_HM_immobilization_prediction_19Feature_0324\ding\zscore_scaler.joblib')
warnings.filterwarnings("ignore", message="missing ScriptRunContext!")

# 设置背景颜色和放大比例的 CSS 代码
background_color = "#f0f8ff"  # 这里可以修改为你想要的背景颜色
zoom_scale = 1.09  # 可以调整这个值来改变放大比例
css = f"""
<style>
    body {{
        background-color: {background_color};
        zoom: {zoom_scale};
    }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# 设置页面标题
st.title('Prediction of the heavy metal environmental risk in fly ash by mechanochemistry')
col1, col2,col3,col4 = st.columns(4)
titles = [
    ("Composition", "#FF5733"),
    ("MC condition", "#33FF57"),
    (" Additives ", "#5733FF"),
    ("HM properties", "#FF33E3")
]


with col1:
    st.markdown(f"<h3 style='color: {titles[0][1]};'>{titles[0][0]}</h3>", unsafe_allow_html=True)
    feature1 = st.number_input(u'$\mathrm{Ca\;(\%)}$', step=0.01, format='%.2f')
    feature2 = st.number_input(u'$\mathrm{Cl\;(\%)}$', step=0.01, format='%.2f')
    feature3 = st.number_input(u'$\mathrm{Si\;(\%)}$', step=0.01, format='%.2f')
    feature4 = st.number_input(u'$\mathrm{Al\;(\%)}$', step=0.01, format='%.2f')
with col2:
    st.markdown(f"<h3 style='color: {titles[1][1]};'>{titles[1][0]}</h3>", unsafe_allow_html=True)
    feature5 = st.number_input(u'$\mathrm{MC\;time\;(h)}$', step=0.5, format='%.1f')
    feature6 = st.number_input(u'$\mathrm{MC\;speed\;(rpm)}$', step=10.0, format='%.1f')
    feature7 = st.number_input(u'$\mathrm{Liquid/Solid}$', step=0.5, format='%.1f')
    feature8 = st.number_input(u'$\mathrm{Ball weight/Weight}$', step=0.5, format='%.1f')
with col3:
    st.markdown(f"<h3 style='color: {titles[2][1]};'>{titles[2][0]}</h3>", unsafe_allow_html=True)
    feature9 = st.number_input(u'$\mathrm{Ca\;additive\;(\%)}$', step=0.5, format='%.1f')
    feature10 = st.number_input(u'$\mathrm{P\;additive\;(\%)}$', step=0.5, format='%.1f')
    feature11 = st.number_input(u'$\mathrm{Ca-P\;additive\;(\%)}$', step=0.5, format='%.1f')
    feature12 = st.number_input(u'$\mathrm{Si-Al\;additive\;(\%)}$', step=0.5, format='%.1f')
with col4:
    st.markdown(f"<h3 style='color: {titles[3][1]};'>{titles[3][0]}</h3>", unsafe_allow_html=True)
    feature13 = st.number_input(u'$\mathrm{Leaching\;method\;pH}$', step=0.01, format='%.2f')
    feature14 = st.number_input(u'$\mathrm{Total\;metal\;(mg/kg)}$', step=0.01, format='%.2f')
    feature15 = st.number_input(u'$\mathrm{Initial\;concentration\;(mg/l)}$', step=0.0001, format='%.4f')
    feature16 = st.number_input(u'$\mathrm{radius\;of\;heavy\;metal\;ions\;(Å)}$', step=0.01, format='%.2f')
feature17 = st.number_input(u'$\mathrm{Electronegativity}$', step=0.01, format='%.2f')

feature = st.number_input(u'$\mathrm{Experimental\;OPTI}$', step=0.01, format='%.4f')
feature_values = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17]

feature_names = ["Ca", "Cl", "Si", "Al", "MC time", "MC speed", "L/S", "BW/W", "Ca-additive", "P-additive",
                     "Ca-P additive",
                     "Si-Al additive", "Leaching pH", "Total metal", "Initial concentration", "Ionic radius",
                     "Electronegativity"]
combined_array = np.array([feature_names, feature_values])

if st.button('Predict'):
    input_data = pd.DataFrame([feature_values], columns=feature_names)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    if prediction[0] <= 0:
        prediction[0] = 0
    if feature != 0:
        Error = abs(float(prediction[0]) - feature)
    else:
        Error = 0
    # final_concentration = (feature15 * (prediction[0] / feature))
    st.success(f'Predicted OPTI: {prediction[0]:.4f}')
    # st.success(f'Predicted concentration: {final_concentration:.2f}(mg/l)')
    st.success(f'Error: {Error:.2f}')
    st.write("Congratulations！！")

