from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Binarizer
from reg_stratified import y_bins, StratifiedKFoldReg
import lightgbm as lgb
import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler


def build_data():
    # load
    original = pd.read_json('../../data/items_ria.json')

    # period to train on
    period_in_month = 6

    # filter
    df = original
    print('Original data shape: {}'.format(df.shape))
    df = df.drop_duplicates(['realty_id'])
    df = df.dropna(axis=0, subset=[
        'price',
        'district_id',
        'building_number_str',
        'floor',
        'quality',
        'created_at',
        'rooms_count',
        'city_id',
        'floors_count',
        'total_square_meters'])
    print('Drop duplicates shape: {}'.format(df.shape))
    df = df[df.created_at >= pd.Timestamp.now() - pd.DateOffset(months=period_in_month)]
    print('Train period shape: {}'.format(df.shape))

    return df


def create_features(df, features):
    df['has_metro'] = df.metro_station_id.notnull()
    features.remove('metro_station_id')
    features.append('has_metro')

    df['is_newbuild'] = df.user_newbuild_id.notnull()
    features.remove('user_newbuild_id')
    features.append('is_newbuild')

    mlb = MultiLabelBinarizer()
    secondaryUtp = pd.DataFrame(mlb.fit_transform(df.secondaryUtp),
                                columns=mlb.classes_,
                                index=df.index)

    cols = list(secondaryUtp.columns)

    cols.remove('Терміново')
    cols.remove('Торг можливий')
    cols.remove('Документи готові')
    cols.remove('Добрі сусіди')
    cols.remove('Вигідна ціна')
    cols.remove('Ексклюзив')
    cols.remove('Елітна квартира')
    cols.remove('Оперативний показ')
    cols.remove('Можлива розстрочка')

    df = pd.concat([df, secondaryUtp[cols]], axis=1)
    features.extend(cols)

    # room features
    df['rooms_count_size'] = df.rooms_count.apply(lambda x: 7 if x > 7 else x)
    df['total_square_meters_by_rooms_count'] = df.total_square_meters / df.rooms_count
    features.remove('rooms_count')
    features.append('rooms_count_size')
    features.append('total_square_meters_by_rooms_count')

    # fix missed values wall_type_uk
    values = df[['wall_type_uk', 'location']].dropna().drop_duplicates(['location'])
    merged = df.merge(values, how='left', on='location')
    df['wall_type_uk'] = merged['wall_type_uk_y']

    # high correlation and missing values
    features.remove('kitchen_square_meters')
    features.remove('living_square_meters')

    # remove imbalance
    features = [x for x in features if x not in [
        'Індивідуальне опалення', 'Багаторівнева', 'Без меблів',
        'Біля аквапарку', 'Біля гір', 'Біля лісу', 'Біля моря', 'Біля річки',
        'Велика кухня', 'Великий балкон', 'Вигідна ціна', 'Високі стелі',
        'Власна тераса', 'Гарний краєвид',
        'Чорнова штукатурка', 'Пентхаус', 'Окремий вхід', 'Панорамні вікна',
        'З підвалом', 'З мансардою', 'З гаражем'
    ]]

    # floor features
    df['last_floor'] = df.floor == df.floors_count
    features.append('last_floor')
    df['first_floor'] = (df.floor == 1) | (df.floor == 0)
    features.append('first_floor')
    df['floor_div_floor_count'] = df.floor / df.floors_count
    features.append('floor_div_floor_count')

    return df, features


def abc():
    df = build_data()

    features = [
        'wall_type_uk',
        'floor',
        'living_square_meters',
        'user_newbuild_id',
        'rooms_count',
        'city_name_uk',
        'kitchen_square_meters',
        'floors_count',
        'total_square_meters',
        'metro_station_id',
        'state_name_uk'
    ]

    df, features = create_features(df, features)

    categorical_features = [
        'floor',
        'wall_type_uk',
        'city_name_uk',
        'state_name_uk',
        'floors_count',
        'has_metro',
        'is_newbuild',
        'Євроремонт',
        'З меблями',
        'З опаленням',
        'З парковкою',
        'З ремонтом',
        'Кухня-студія',
        'Можна з тваринами',
        'Новий ремонт',
        'Поруч з метро',
        'Поруч з парком',
        'Поруч зі школою',
        'Поруч із дитячим садком',
        'Преміум клас',
        'Престижний район',
        'Світла і простора',
        'Спокійний район',
        'Сучасний дизайн',
        'Територія під охороною',
        'Тихий двір',
        'У центрі',
        'rooms_count_size',
        'last_floor',
        'first_floor'
    ]

    df[categorical_features] = df[categorical_features].astype('category')

    print(df[features].iloc[0].to_json(force_ascii=False))


def train_model():

    df = build_data()

    features = [
        'wall_type_uk',
        'floor',
        'living_square_meters',
        'user_newbuild_id',
        'rooms_count',
        'city_name_uk',
        'kitchen_square_meters',
        'floors_count',
        'total_square_meters',
        'metro_station_id',
        'state_name_uk'
    ]

    df, features = create_features(df, features)

    categorical_features = [
        'floor',
        'wall_type_uk',
        'city_name_uk',
        'state_name_uk',
        'floors_count',
        'has_metro',
        'is_newbuild',
        'Євроремонт',
        'З меблями',
        'З опаленням',
        'З парковкою',
        'З ремонтом',
        'Кухня-студія',
        'Можна з тваринами',
        'Новий ремонт',
        'Поруч з метро',
        'Поруч з парком',
        'Поруч зі школою',
        'Поруч із дитячим садком',
        'Преміум клас',
        'Престижний район',
        'Світла і простора',
        'Спокійний район',
        'Сучасний дизайн',
        'Територія під охороною',
        'Тихий двір',
        'У центрі',
        'rooms_count_size',
        'last_floor',
        'first_floor'
    ]

    df[categorical_features] = df[categorical_features].astype('category')

    # preparing price in single currency
    df = df[df.priceArr.map(lambda arr: len(arr.keys()) > 0)]
    price = df.priceArr.map(lambda arr: int(arr['1'].replace(' ', '')))

    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(df[features], price,
                                                        test_size=0.2, random_state=42, stratify=y_bins(price, 100))

    # folds
    skf = StratifiedKFoldReg(shuffle=True, random_state=42)

    # transform y
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # scorer
    rmse = make_scorer(lambda y1, y2: mean_squared_error(y1, y2, squared=False), greater_is_better=False)

    # transform X
    column_transformer = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_features),
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    train_data = column_transformer.fit_transform(X_train)
    test_data = column_transformer.transform(X_test)

    # baselines
    print('mean')
    print(mean_squared_error(np.full(y_train_log.size, y_train_log.mean()), y_train_log, squared=False))
    print(mean_squared_error(np.full(y_test_log.size, y_train_log.mean()), y_test_log, squared=False))

    print('median')
    print(mean_squared_error(np.full(y_train_log.size, y_train_log.median()), y_train_log, squared=False))
    print(mean_squared_error(np.full(y_test.size, y_train_log.median()), y_test_log, squared=False))

    print('mode')
    print(mean_squared_error(np.full(y_train_log.size, y_train_log.mode()), y_train_log, squared=False))
    print(mean_squared_error(np.full(y_test_log.size, y_train_log.mode()), y_test_log, squared=False))

    baseline_model = LinearRegression()
    print(baseline_model)
    print(cross_val_score(baseline_model, train_data, y_train_log, cv=skf, scoring=rmse))

    baseline_model = DecisionTreeRegressor()
    print(baseline_model)
    print(cross_val_score(baseline_model, train_data, y_train_log, cv=skf, scoring=rmse))

    # training
    parameters = {

        # default
        'objective': 'regression',
        "learning_rate": 0.03,
        "n_jobs": -1,
        "seed": 42,
        "metric": "rmse",

        # #regularization
        "subsample_freq": 1,
        "min_child_samples": None,

        'subsample': 0.85,
        'colsample_bytree': 0.7,
        'min_data_per_group': 10,
        'min_data_in_leaf': 2,
        'cat_smooth': 5
    }

    # param_grid = {
    #     "colsample_bytree": [0.75, 0.7],
    #     "subsample": [0.87, 0.85],
    #     "learning_rate": [0.05, 0.03, 0.01],

    #     "min_data_in_leaf": [2, 3, 4, 5],
    #     "min_data_per_group": [8, 9, 10, 12],
    #     "cat_smooth": [3, 5, 8, 9]
    # }

    # # Generate all combinations of hyperparameters
    # param_combinations = list(ParameterSampler(param_grid, n_iter=50, random_state=np.random.RandomState(42)))

    # cv_results = []

    # for params in param_combinations:

    #     params.update(parameters)

    #     n_rounds = 100000
    #     lgb_train = lgb.Dataset(X_train, label=y_train_log)
    #     lgb_result = lgb.cv(params, lgb_train, n_rounds, folds=skf,
    #                         early_stopping_rounds=100, verbose_eval=0)

    #     cv_results.append((params, lgb_result['rmse-mean'][-1]))

    #     cv_results.sort(key=lambda x: x[1], reverse=False)
    #     best_params, best_score = cv_results[0]

    #     print("Best parameters found:", best_params)
    #     print("Best score:", best_score)

    n_rounds = 100000
    lgb_train = lgb.Dataset(X_train, label=y_train_log)
    lgb_result = lgb.cv(parameters, lgb_train, n_rounds, folds=skf,
                        early_stopping_rounds=100, verbose_eval=100)

    num_boost_round = len(lgb_result['rmse-mean'])
    lgb_model = lgb.train(parameters, lgb_train, num_boost_round=num_boost_round)

    train_predictions = lgb_model.predict(X_train)

    print(mean_squared_error(train_predictions, y_train_log, squared=False))
    print(mean_squared_error(lgb_model.predict(X_test), y_test_log, squared=False))

    full_dataset = lgb.Dataset(df[features], label=np.log1p(price))
    final_model = lgb.train(parameters, full_dataset, num_boost_round=num_boost_round)
    final_model.save_model("../../model/model.pkl", num_iteration=num_boost_round)


def main():
    abc()


if __name__ == '__main__':
    main()