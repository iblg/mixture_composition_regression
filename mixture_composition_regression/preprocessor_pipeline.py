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

    :param lbounds: tuple, default (900, 3200).
    The lower and upper bounds on the wavelength.
    :param m: Mixture object.

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
    for sample in da.coords['name'].values:
        selection = da.sel({'name': sample}).dropna(dim='l', how='all')
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
    if ycol is None:
        pass
    else:
        y = y[:, ycol]
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
