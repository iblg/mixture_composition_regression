# from typing import Dict
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import median_absolute_error
from sklearn.metrics import PredictionErrorDisplay

import numpy as np
import scipy as sp

import mixture_composition_regression.mixture
from mixture_composition_regression.sample import Sample
from mixture_composition_regression.mixture import Mixture
from mixture_composition_regression.import_spectrum import clean_data
import matplotlib.pyplot as plt
import xarray as xr


def import_training_set():
    cp = {'name': ['water', 'dipa', 'nacl'],
          'mw': [18.015, 101.19, 58.44],
          'nu': [1, 1, 2]}

    # 03-03-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-03/2023-03-03.csv'
    df = clean_data(file)
    water1 = Sample('water1', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])
    dipa1 = Sample('dipa1', df, 4, 5, chem_properties=cp, w=[0., 1., 0.])

    # 03-07-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-07/2023-03-07.csv'
    df = clean_data(file)
    water2 = Sample('water2', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])
    dipa2 = Sample('dipa2', df, 4, 5, chem_properties=cp, w=[0., 1., 0.])
    dipa_w1 = Sample('dipa_w1', df, 6, 7, chem_properties=cp,
                     w=[0.0910 / (0.0910 + 0.9474), 0.9474 / (0.0910 + 0.9474), 0.])
    dipa_w2 = Sample('dipa_w2', df, 8, 9, chem_properties=cp,
                     w=[0.1510 / (0.1510 + 1.0358), 1.0358 / (0.1510 + 1.0358), 0.])

    # 03-09-2023
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-09/2023-03-09.csv'
    df = clean_data(file)
    dipa_w1a = Sample('dipa_w1a', df, 0, 1, chem_properties=cp,
                      w=[0.0910 / (0.0910 + 0.9474), 0.9474 / (0.0910 + 0.9474), 0.])
    dipa_w2a = Sample('dipa_w2a', df, 2, 3, chem_properties=cp,
                      w=[0.1510 / (0.1510 + 1.0358), 1.0358 / (0.1510 + 1.0358), 0.])
    dipa_w3 = Sample('dipa_w3', df, 4, 5, chem_properties=cp,
                     w=[0.0382 / (0.0382 + 0.8671), 0.8671 / (0.0382 + 0.8671), 0.])
    dipa_w4 = Sample('dipa_w4', df, 6, 7, chem_properties=cp,
                     w=[0.3690 / (0.3690 + 1.1550), 1.1550 / (0.3690 + 1.1550), 0.])

    # 03-22-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-22/2023-03-22.csv'
    df = clean_data(file)
    water3 = Sample('water3', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])

    five_M = Sample('5M', df, 4, 5, chem_properties=cp, w=[1. - 0.2470, 0., 0.2470])
    five_M_2 = Sample('5M_2', df, 6, 7, chem_properties=cp, w=[1. - 0.2470, 0., 0.2470])
    two_M = Sample('2M', df, 8, 9, chem_properties=cp, w=[1. - 0.1087, 0., 0.1087])
    two_M_2 = Sample('2M_2', df, 10, 11, chem_properties=cp, w=[1. - 0.1087, 0., 0.1087])
    four_M = Sample('4M', df, 12, 13, chem_properties=cp, w=[1. - 0.2036, 0., 0.2036])
    four_M_2 = Sample('4M_2', df, 14, 15, chem_properties=cp, w=[1. - 0.2036, 0., 0.2036])

    m1 = Mixture([water1, dipa1, water2, water3, dipa2, dipa_w1, dipa_w1a, dipa_w2, dipa_w2a, dipa_w3, dipa_w4,
                  five_M, five_M_2, two_M, two_M_2, four_M, four_M_2])
    return m1


def plot_single(m):
    fig, ax = plt.subplots()
    data = m.da.sel(name='water1')
    ax.plot(data.l, data.values, '.')
    plt.show()
    return


def get_Xy(m, lbounds=None):
    # X = []
    y = []
    da = m.da

    if lbounds is not None:
        bds = (da.l.values > lbounds[0]) & (da.l.values < lbounds[1])
        da = da.where(bds).dropna(dim='l')

    chems = da.coords
    del chems['l']
    del chems['name']

    first = 0
    # y = np.array([[]])
    for val in da.coords['name'].values:
        selection = da.sel({'name': val}).dropna(dim='l', how='all')

        first_chem = 0
        for chem in chems:
            if first_chem == 0:
                comp = selection.coords[chem]
                first_chem += 1
            else:
                comp = np.append(comp, selection.coords[chem].values)

        x = selection.values.reshape(-1, 1)
        comp = comp.reshape(-1, 1)

        if first == 0:
            X = x
            y = comp
            first += 1
        else:
            X = np.append(X, x, axis=1)
            y = np.append(y, comp, axis=1)

    X, y = X.T, y.T
    return y, X


def get_preprocessor():
    categorical_columns = []
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop="if_binary"), categorical_columns),
        remainder="passthrough",
        verbose_feature_names_out=False,  # avoid to prepend the preprocessor names
    )

    return preprocessor


def get_pipeline(preprocessor, regr=None, func=None, inverse_func=None):
    if regr is None:
        regr = Ridge(alpha=1e-8)

    if func is None:
        f = identity
    else:
        f = func

    if inverse_func is None:
        f_inv = identity
    else:
        f_inv = inverse_func

    model = make_pipeline(
        preprocessor,
        TransformedTargetRegressor(regressor=regr, func=f, inverse_func=f_inv)
    )
    return model


def identity(x):
    return x


def plot_mae(y_test, y_train, y_pred, mae_test, mae_train):
    scores = {
        "MedAE on training set": f"{mae_train:.4f}",
        "MedAE on testing set": f"{mae_test:.4f}",
    }

    _, ax = plt.subplots(figsize=(5, 5))
    display = PredictionErrorDisplay.from_predictions(
        y_test, y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
    )
    ax.set_title("Ridge model, small regularization")
    for name, score in scores.items():
        ax.plot([], [], " ", label=f"{name}: {score}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    return


def main():
    preprocessor = get_preprocessor()
    # model = get_pipeline(preprocessor, regr=Ridge(alpha=10**-7))
    model = get_pipeline(preprocessor, regr=RidgeCV(alphas=np.logspace(-2, 5, 11)))
    # model = get_pipeline(preprocessor, regr=SVR(gamma='scale', epsilon=0.0001))
    m1 = import_training_set()
    plot_single(m1)
    y, X = get_Xy(m1, lbounds=[1200, 1700])
    y = y[:, 0]  # only regress water
    # y = y[:, 1]  # only regress dipa
    # y = y[:, 2]  # only regress nacl
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_train = median_absolute_error(y_train, model.predict(X_train))
    mae_test = median_absolute_error(y_test, y_pred)
    plot_mae(y_test, y_train, y_pred, mae_test, mae_train)

    return


if __name__ == '__main__':
    main()
