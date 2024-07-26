"""
El conjunto de datos de previsión de la demanda está diseñado para proporcionar una base sólida para analizar y predecir la demanda 
futura en diversas industrias y productos. Este conjunto de datos es especialmente valioso para empresas, analistas e investigadores que 
se centran en comprender la dinámica del mercado, optimizar la gestión de inventarios y mejorar la eficiencia de la cadena de suministro."""

#Libraries
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import missingno as msno
from matplotlib import pyplot as plt
plt.style.use('dark_background')
from lightgbm import early_stopping, log_evaluation

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

lgb.__version__

#Cargar datos
train = pd.read_csv('demanda/train.csv', parse_dates=['date'])
test = pd.read_csv('demanda/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('demanda/sample_submission.csv')

df = pd.concat([train, test], sort=False)
#EDA
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

print(df.head(5))

print(df.tail(5))

df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T

df[["store"]].nunique()

df[["item"]].nunique()

df.groupby(["store"])["item"].nunique()

df.groupby(["store", "item"]).agg({"sales": ["sum"]})

df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

#Ingeniería de características
def create_date_features(dataframe:pd.DataFrame) -> pd.DataFrame:
    dataframe['month'] = dataframe.date.dt.month
    dataframe['day_of_month'] = dataframe.date.dt.day
    dataframe['day_of_year'] = dataframe.date.dt.dayofyear
    dataframe['week_of_year'] = dataframe.date.dt.isocalendar().week
    dataframe['day_of_week'] = dataframe.date.dt.dayofweek
    dataframe['year'] = dataframe.date.dt.year
    dataframe["is_wknd"] = dataframe.date.dt.weekday // 4
    dataframe['is_month_start'] = dataframe.date.dt.is_month_start.astype(int)
    dataframe['is_month_end'] = dataframe.date.dt.is_month_end.astype(int)
    return dataframe

df = create_date_features(df)
df.head()

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})

#Ruido aleatorio
def random_noise(dataframe:pd.DataFrame) -> np.ndarray:
    return np.random.normal(scale=1.6, size=(len(dataframe),))
# Funciones de retraso/desplazadas
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})

df.groupby(["store", "item"])['sales'].head()

df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))

#Características de retraso
def lag_features(dataframe:pd.DataFrame, lags:list) -> pd.DataFrame:
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


lags = [91, 98, 105, 112, 119, 126, 182, 364, 546, 728]  # since we want to forecast 3 month periods.
df = lag_features(df, lags)
df

#Características medias rodantes
# No recomendado (cambio primero)
'''
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})
'''
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})

def roll_mean_features(dataframe:pd.DataFrame, windows:list) -> pd.DataFrame:
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

windows = [365, 546]  #Gama  1 y 1,5 años
df = roll_mean_features(df, windows)  
df.head()

#Características medias ponderadas exponencialmente
# media ponderada -> para datos cercanos
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})

def ewm_features(dataframe:pd.DataFrame, alphas:list, lags:list) -> None:
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
df

# Codificación en caliente
cat_cols = ['store', 'item', 'day_of_week', 'month']
df = pd.get_dummies(df, columns=cat_cols)
df

#Convertir ventas en registro (1+ventas)
# Conversión de variable dependiente
df['sales'] = np.log1p(df["sales"].values)
df.head()

#Modelo
#Función de costo personalizada
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False
# Conjuntos de validación basados ​​en tiempo
train = df.loc[(df["date"] < "2017-01-01"), :]
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]
print(train.shape, val.shape)

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

print(Y_train.shape, X_train.shape, Y_val.shape, X_val.shape)

#Modelo de series temporales con LightGBM
lgb_params = {'num_leaves': 10,               # Número máximo de hojas
              'learning_rate': 0.02,          # Tasa de aprendizaje (tasa de contracción, eta)
              'feature_fraction': 0.8,        # Fracción de característica (característica del subespacio aleatorio de rf)
              'max_depth': 5,                 # Máxima profundidad
              'verbose': 0,                   # Detallado (informe)
              'num_boost_round': 1000,        # n_estimators(número de iteraciones de impulso) al menos 10000-15000
              'early_stopping_rounds': 200,   # Parada temprana
              'nthread': -1}                  # num_thread, nthread, nthreads, n_jobs
lgb_train = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgb_val = lgb.Dataset(data=X_val, label=Y_val, reference=lgb_train, feature_name=cols)

model = lgb.train(lgb_params, lgb_train,
                  valid_sets=[lgb_train, lgb_val],
                  num_boost_round=lgb_params['num_boost_round'],
                  callbacks=[early_stopping(stopping_rounds=lgb_params['early_stopping_rounds']),
                             log_evaluation()],
                  feval=lgbm_smape)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))

#Importancia del modelo
def plot_lgb_importances(model, plot:bool=False, num:int=10) -> pd.DataFrame:
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp


plot_lgb_importances(model, num=30, plot=True)

feat_imp = plot_lgb_importances(model, num=200)
importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)

#Modelo final
train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)
#Archivo de envío
test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)
submission_df.to_csv("submission_demand.csv", index=False)


