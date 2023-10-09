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


def get_chemlist(da):
    """
    Returns a list of the chemical species in a mixture.

    :param da: xarray.DataArray object
    The Mixture.da object being interrogated.

    :return: chems
    """
    chems = da.coords
    del chems['x']
    del chems['name']
    return chems


def get_Xy(m, lbounds, ycol=None):
    """
    Processes a mixture to get X (data), y (target variable) information to send to the model.

    :param m: Mixture object
    The mixture object that will be used to develop a model.

    :param lbounds: tuple, default (900, 3200).
    The lower and upper bounds on the wavelength.

    :param ycol: int
    The column index for the target variable

    :return:
    :param y: numpy Array
    Contains the target variable.
    :param X: numpy Array
    Contains the training variables.

    """

    da = m.da

    bds = (da.x.values > lbounds[0]) & (da.x.values < lbounds[1])
    da = da.where(bds).dropna(dim='x')

    # get list of chemicals in the mixture
    chems = get_chemlist(da)

    first = 0
    for i in da.coords['name'].values:
        selection = da.sel({'name': i}).dropna(dim='x', how='all')

        first_chem = 0
        for chem in chems:  # get the composition of the mixture.
            if first_chem == 0:
                composition = selection.coords[chem]
                first_chem += 1
            else:
                composition = np.append(composition, selection.coords[chem].values)

        x = selection.values.reshape(-1, 1)
        x = pd.DataFrame(x, )

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
    """
    Processes a mixture to get X, the dependent variables, and y, the target variable. Returns y, X.

    :param m: mixture_composition_regression.Mixture object.
    The mixture forming the training set for the model.

    :param lbounds: list
    List of lower and upper bounds on wavelength range desired.

    :param target_chem: str or int, default None.
    If target_chem is int, column index for target chemical data in sample.w
    If target_chem is str, column name for target chemical data in sample.w

    :return:
    """

    for i, sample in enumerate(m.samples):
        if i == 0:
            y = sample.w.rename(sample.name)
            X = sample.a.loc[lbounds[0]:lbounds[1]].rename(sample.name)
        else:
            y0 = sample.w.rename(sample.name)
            y = pd.concat([y, y0], axis=1)

            x = sample.a.loc[lbounds[0]:lbounds[1]].rename(sample.name)
            X = pd.concat([X, x], axis=1)

    X = X.T  # transpose the X array

    # Find target variable.
    if target_chem is None:
        pass
    elif type(target_chem) is int:
        y = y.iloc[target_chem]
    elif type(target_chem) is str:
        y = y.loc[target_chem]

    return y, X


def get_preprocessor(cat_columns=[]):
    """
    :param cat_columns: list of str
    List of column names describing categorical data.
    :return:
    """
    categorical_columns = cat_columns
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop="if_binary"), categorical_columns),
        remainder="passthrough",
        verbose_feature_names_out=False,  # avoid to prepend the preprocessor names
    )
    return preprocessor


def get_pipeline(preprocessor, regr=None, func=None, inverse_func=None):
    """

    :param preprocessor:

    :param regr: regressor

    :param func: function, default None.
    Function with which to transform data. If None, no transformation is applied to the data.
    If another function is provided, such as np.log10, the data is transformed before being regressed, then re-transformed
    using the inverse function inverse_func

    :param inverse_func: function, default None.
    The inverse function used to transform back from func. If func is provided, an inverse_func must also be provided.

    :return:
    """

    # check some edge cases
    if func and inverse_func:
        pass
    elif not func and not inverse_func:
        pass
    elif not inverse_func:
        print('No inverse_func provided to get_pipeline')
    elif not func:
        print('No func provided to get_pipeline; inverse_func provided.')

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
    """
    Simple function that returns unaltered argument.
    :param x:
    :return:
    """
    return x


def plot_metric(y_test, y_train, y_pred, metric_label, metric_test, metric_train,
                filepath: str = None,
                wl_window: list = None,
                display: bool = False):
    """

    :param y_test:
    Target variable data for model evaluation.

    :param y_train:
    Target variable data used to train model.

    :param y_pred:
    Model predictions for y_train.

    :param metric_label: str
    String used to name the goodness of fit metric used to evaluate the model.
    Common examples might be r'$R^{2}$' or r'$\chi^2$'.

    :param metric_test: float
    Result for goodness of fit on testing data set.

    :param metric_train: float
    Result for goodness of fit on training data set.

    :param filepath: str, default None
    The filepath to save a plot of results. If None, no plot is saved.

    :param wl_window: str, default None
    The wavelength window used to train the model.
    If None, no wavelength is printed on the plot.
    If not None, the wavelength interval is printed on the plot.

    :param display: bool, default False
    If True, a plot of results will be shown.

    :return:
    """

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

    if display:
        plt.show()

    if filepath:
        pass
    else:
        fig.savefig(filepath + '.png', dpi=400)
    return fig, ax
