
# Импортируем необходимые библиотеки для нашего приложения

import pathlib              #для задания относительного пути к файлам
from pathlib import Path    #для задания относительного пути к файлам


import pandas as pd
import numpy as np
import joblib               # для сохранения/загрузки моделей машинного обучения


from sklearn.linear_model import LinearRegression
from tensorflow import keras
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler


from flask import Flask, request, render_template  # для работы приложения


# Определим пути к предварительно подготовленной модели и нормализаторам
dir_path = pathlib.Path.cwd()
paht_model_ann = Path(dir_path.parents[0], 'domain', 'models_saved', 'ann', 'recommendation', 'my_model_1.h5')
paht_norm_y_ann = Path(dir_path.parents[0], 'domain', 'models_saved', 'ann', 'recommendation', 'norm_minmax_y.joblib')
paht_norm_x_ann = Path(dir_path.parents[0], 'domain', 'models_saved', 'ann', 'recommendation', 'norm_minmax_x.joblib')


paht_model_lr = Path(dir_path.parents[0], 'domain', 'models_saved', 'prediction_of_properties', 'lr_joblib_model.joblib')
paht_norm_y_lr = Path(dir_path.parents[0], 'domain', 'models_saved', 'prediction_of_properties', 'stand_y.joblib')
paht_norm_x_lr = Path(dir_path.parents[0], 'domain', 'models_saved', 'prediction_of_properties', 'stand_x.joblib')


# Загрузим обученные модели
model_lr = LinearRegression()
model_lr = joblib.load(paht_model_lr)
model_ann = keras.models.load_model(paht_model_ann)

# Загрузим нормализаторы
norm_y_lr = StandardScaler()
norm_x_lr = StandardScaler()
norm_y_lr = joblib.load(paht_norm_y_lr)
norm_x_lr = joblib.load(paht_norm_x_lr)

norm_y_ann = MinMaxScaler()
norm_x_ann = MinMaxScaler()
norm_y_ann = joblib.load(paht_norm_y_ann)
norm_x_ann = joblib.load(paht_norm_x_ann)


# наше приложение
app = Flask(__name__, static_folder="/")


def calculate_recomendation(param):
    param = np.array(param)
    param = pd.DataFrame(param).T
    param = norm_x_ann.transform(param)                     # нормализация входных параметров

    prediction = model_ann.predict(param)                   # предсказание моделью
    prediction = norm_y_ann.inverse_transform(prediction)   # денормализация предсказанного значения
    return prediction


def calculate_prediction(param):
    param = np.array(param)
    param = pd.DataFrame(param).T
    param = norm_x_lr.transform(param)                     # нормализация входных параметров

    prediction = model_lr.predict(param)                   # предсказание моделью

    prediction = pd.DataFrame(prediction)

    prediction = norm_y_lr.inverse_transform(prediction)    # денормализация предсказанного значения

    return prediction




@app.route('/', methods=['get', 'post'])
def index():

    return render_template('index.html')



# страница рекомендации соотношения матрица-наполнитель
@app.route('/calculation_1', methods=['post', 'get'])
def calculation_1():
    param_lst = []
    message = ''
    alert = False
    if request.method == 'POST':

        # получим данные из наших форм и кладем их в список, который затем передадим функции calculate_recomendation
        for i in range(1, 13, 1):
            param = request.form.get(f'param{i}')
            param_lst.append(param)

        for i in range(0, 12, 1):
            if param_lst[i] == '':
                alert = True
            else:
                try:
                    param_lst[i] = float(param_lst[i])
                except ValueError:
                    alert = True
                    break

        if alert == False:
            message = calculate_recomendation(param_lst)
            message = round(message[0][0], 4)
        else:
            message = ''


    return render_template('calculation_1.html', message = message, value = param_lst, alert = alert)


# страница предсказания прочности при растяжении
@app.route('/calculation_2', methods=['post', 'get'])
def calculation_2():
    param_lst = []
    message = ''
    alert = False
    if request.method == 'POST':

        # получим данные из наших форм и кладем их в список, который затем передадим функции calculate_prediction
        for i in range(1, 13, 1):
            param = request.form.get(f'param{i}')
            param_lst.append(param)

        for i in range(0, 12, 1):
            if param_lst[i] == '':
                alert = True
            else:
                try:
                    param_lst[i] = float(param_lst[i])
                except ValueError:
                    alert = True
                    break

        if alert == False:
            message = calculate_prediction(param_lst)
            message = round(message[0][0], 4)
        else:
            message = ''

    return render_template('calculation_2.html', message = message, value = param_lst, alert = alert)



if __name__ == "__main__":
    app.run(debug=True)