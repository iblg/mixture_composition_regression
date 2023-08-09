import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import make_column_transformer
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import Ridge

import numpy as np
import matplotlib.pyplot as plt


def get_Xy(m, lbounds, ycol=None):
    """
    :param m: Mixture object.

    :param lbounds: tuple, default (900, 3200).
    The lower and upper bounds on the wavelength.

    :return:
    :param y: numpy array
    Contains the target variable.
    :param X: numpy array
    Contains the training variables.
    """

    da = m.da

    bds = (da.l.values > lbounds[0]) & (da.l.values < lbounds[1])
    da = da.where(bds).dropna(dim='l')

    chems = da.coords
    del chems['l']
    del chems['name']

    first = 0
    for i in da.coords['name'].values:
        selection = da.sel({'name': i}).dropna(dim='l', how='all')
        first_chem = 0

        for chem in chems: # get the composition of the mixture.
            if first_chem == 0:
                composition = selection.coords[chem]
                first_chem += 1
            else:
                composition = np.append(composition, selection.coords[chem].values)

        x = selection.values.reshape(-1, 1)
        x = pd.DataFrame(x,)
        composition = composition.reshape(-1, 1)

        if first == 0:
            X = x
            y = composition
            first += 1
        else:
            X = np.append(X, x, axis=1)
            y = np.append(y, composition, axis=1)

    if ycol is None:
        pass
    else:
        y = y[:, ycol]
    return y, X

def get_Xy_2(m, lbounds, target_chem=None):

    for i, sample in enumerate(m.samples):
        if i == 0:
            y = sample.w.rename(sample.name)
            X = sample.a.loc[lbounds[0]:lbounds[1]].rename(sample.name)
        else:
            y0 = sample.w.rename(sample.name)
            y = pd.concat([y, y0], axis=1)

            x = sample.a.loc[lbounds[0]:lbounds[1]].rename(sample.name)
            X = pd.concat([X, x], axis=1)

    if target_chem is None:
        pass
    elif type(target_chem) is int:
        y = y.iloc[target_chem]
    elif type(target_chem) is str:
        y = y.loc[target_chem]

    X = X.T

    return y, X


def get_preprocessor(cat_columns = []):
    categorical_columns = cat_columns
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


def plot_metric(y_test, y_train, y_pred, metric_label, metric_test, metric_train,
                savefile=None,
                wl_window=None,
                display=False):
    scores = {
        '{} on training set'.format(metric_label): '{:.4f}'.format(metric_train),
        '{} on testing set'.format(metric_label): '{:.4f}'.format(metric_test),
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    PredictionErrorDisplay.from_predictions(
        y_test, y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
    )
    # ax.set_title("Ridge model, small regularization")
    if wl_window is None:
        pass
    else:
        ax.text(0.95, 0.1, r'$\lambda_{\mathrm{min}} =$' + '{:3.1f}'.format(wl_window[0]), transform=ax.transAxes,
                ha='right', va='bottom')
        ax.text(0.95, 0.05, r'$\lambda_{\mathrm{max}} =$' + '{:3.1f}'.format(wl_window[1]), transform=ax.transAxes,
                ha='right', va='bottom')

    for name, score in scores.items():
        ax.plot([], [], " ", label=f"{name}: {score}")
    ax.legend(loc="upper left")
    plt.tight_layout()

    if display is True:
        plt.show()

    if savefile is None:
        pass
    else:
        fig.savefig(savefile + '.png', dpi=400)
    return fig, ax
