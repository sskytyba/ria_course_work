import logging

import numpy as np
import pandas as pd
import streamlit as st
import json

from src.deploy.api import Healthcheck, FlatRegressorApi
from src.predictor import FlatRegressor
from src.train.train import data

logging.getLogger("transformers").setLevel(logging.ERROR)

regressor = FlatRegressor.load("model/")

st.title("Predict flat price")

city_name_uk = st.selectbox("Місто", ('Івано-Франківськ',
  'Ірпінь',
  'Агрономічне',
  'Бориспіль',
  'Бровари',
  'Буча',
  'Біла Церква',
  'Винники',
  'Вишгород',
  'Вишневе',
  'Вінниця',
  'Гатне',
  'Гостомель',
  'Дніпро',
  'Житомир',
  'Жуляни',
  'Запоріжжя',
  "Кам'янське",
  'Київ',
  'Козин',
  'Коцюбинське',
  'Кременчук',
  'Кривий Ріг',
  'Крихівці',
  'Кропивницький',
  'Крюківщина',
  'Луцьк',
  'Львів',
  'Миколаїв',
  'Мукачево',
  'Новосілки',
  'Одеса',
  'Полтава',
  'Рівне',
  'Святопетрівське',
  'Слобожанське',
  'Сокільники',
  'Стрий',
  'Суми',
  'Тернопіль',
  'Трускавець',
  'Угорники',
  'Ужгород',
  'Харків',
  'Херсон',
  'Хмельницький',
  'Чабани',
  'Чайки',
  'Черкаси',
  'Чернівці',
  'Чернігів'), index=18)
state_name_uk = st.selectbox('Область', ('Івано-Франківська',
  'Волинська',
  'Вінницька',
  'Дніпропетровська',
  'Житомирська',
  'Закарпатська',
  'Запорізька',
  'Київська',
  'Кіровоградська',
  'Львівська',
  'Миколаївська',
  'Одеська',
  'Полтавська',
  'Рівненська',
  'Сумська',
  'Тернопільська',
  'Харківська',
  'Херсонська',
  'Хмельницька',
  'Черкаська',
  'Чернівецька',
  'Чернігівська'), index=7)
floor = st.number_input("Поверх", min_value=0, max_value=40, value=5, step=1)
floors_count = st.number_input("Всього поверхів", min_value=1, max_value=40, value=10, step=1)
rooms_count = st.number_input("Кількість кімнат", min_value=1, max_value=7, value=3, step=1)
total_square_meters = st.number_input("Площа", value=70)
wall_type_uk = st.selectbox("Тип стін", ('СІП',
  'армований залізобетон',
  'блочно-цегляний',
  'бутовий камінь',
  'газобетон',
  'газоблок',
  'дерево та цегла',
  'залізобетон',
  'збірний залізобетон',
  'каркасний',
  "каркасно-кам'яний",
  'каркасно-панельний',
  'керамзітобетон',
  'керамічна цегла',
  'керамічний блок',
  'моноліт',
  'монолітний залізобетон',
  'монолітно-блоковий',
  'монолітно-каркасний',
  'монолітно-цегляний',
  'облицювальна цегла',
  'панель',
  'полістиролбетон',
  'піноблок',
  'ракушняк',
  'силікатна цегла',
  'цегла',
  'червона цегла',
  'інкерманський камінь'), index=26)
has_metro = st.checkbox("Метро")
is_newbuild = st.checkbox("Новобудова")
all_options = ('З_меблями',
               'З_парковкою',
               'З_ремонтом',
               'Кухня-студія',
               'Можна_з_тваринами',
               'Поруч_з_метро',
               'Поруч_з_парком',
               'Поруч_зі_школою',
               'Поруч_із_дитячим_садком')
options = st.multiselect('Опції', all_options)

sample = {
    "wall_type_uk": wall_type_uk,
    "floor": floor,
    "city_name_uk": city_name_uk,
    "floors_count": floors_count,
    "total_square_meters": total_square_meters,
    "state_name_uk": state_name_uk,
    "has_metro": has_metro,
    "is_newbuild": is_newbuild,
    "rooms_count_size": rooms_count,
    'total_square_meters_by_rooms_count': (1.0 * total_square_meters) / rooms_count,
    'last_floor': floor == floors_count,
    'first_floor': (floor == 1) | (floor == 0),
    'floor_div_floor_count': (1.0 * floor) / floors_count
}

for o in all_options:
    sample[o] = 1 if (o in options) else 0

res = st.button("Calc!")

if res:
    predictions = regressor.predict([sample])
    st.success(str(round(predictions[0])) + ' $')


percent = state_name_uk = st.selectbox('Відсоток', (10, 20, 30, 40, 50, 60, 70, 80, 90, 100), index=3)
show = st.button("Show")

if show:
    df = pd.read_csv('data/items_ria_with_prediction.csv')
    df = df[df.diff_per >= percent]
    df = df.sort_values(by=['diff_per'])

    print(df.shape)

    features = ['wall_type_uk', 'floor', 'city_name_uk', 'floors_count', 'total_square_meters', 'state_name_uk',
                'has_metro', 'is_newbuild', 'З меблями', 'З парковкою', 'З ремонтом', 'Кухня-студія', 'Можна з тваринами',
                'Поруч з метро', 'Поруч з парком', 'Поруч зі школою', 'Поруч із дитячим садком', 'rooms_count_size',
                'total_square_meters_by_rooms_count', 'last_floor', 'first_floor', 'floor_div_floor_count']

    res = df[['realty_id', 'real_price', 'predict_price', 'diff_per'] + features]
    st.dataframe(res[res.real_price > res.predict_price])
    st.dataframe(res[res.real_price < res.predict_price])