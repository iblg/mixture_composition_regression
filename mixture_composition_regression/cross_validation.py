import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
# from ib_mpl_stylesheet.ib_mpl_stylesheet import ib_mpl_style

import numpy as np

import mixture_composition_regression.mixture
from mixture_composition_regression.examples.import_training_set import import_training_set
from mixture_composition_regression.preprocessor_pipeline import *


# They need to be able to try different models, different params within those models, and different wavelength ranges

def cv_on_model_and_wavelength(m: mixture_composition_regression.mixture.Mixture,
                               nwindows: list,
                               models: list,
                               l_bounds: tuple,
                               target_chem=None,
                               tts_test_size: float = None,
                               tts_random_state: int = None,
                               tolerance: float = 0.01,
                               metric: sklearn.metrics = None,
                               metric_label: str = None,
                               test_data: mixture_composition_regression.mixture.Mixture = None,
                               plot_comparison: bool = False,
                               plot_comparison_savefile: str = None):
    if metric is None:
        metric = mean_squared_error
    else:
        pass

    best_score = 10 ** 20

    viable_models = []
    for n in nwindows:
        print('Running analysis splitting interval into {} windows.'.format(n))
        if l_bounds:
            pass
        else:
            l_bounds = (min(m.samples[0].l), max(m.samples[0].l))

        wl = get_window_list(l_bounds[0], l_bounds[1], nwindows=n)

        for idx, model in enumerate(models):
            print('Running analysis on', model.estimator)

            for l_window in wl:
                # l_window[0] -= 1E-5  # this stops the bottom-most interval from being shorter than the others.

                if test_data is None:
                    y, X = get_Xy_2(m, lbounds=l_window, target_chem=target_chem)  # get y, X data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts_test_size,
                                                                        random_state=tts_random_state)
                else:
                    y_train, X_train = get_Xy_2(m, lbounds=l_window, target_chem=target_chem)  # get y, X data
                    y_test, X_test = get_Xy_2(test_data, lbounds=l_window, target_chem=target_chem)

                model_instance = model.fit(X_train,
                                           y_train)  # model instance is the model with optimized params by gridsearch CV

                # Evaluate the model

                y_pred = model_instance.predict(X_test)
                train_eval = metric(y_train, model_instance.predict(X_train))
                score = metric(y_test, y_pred)

                if score < tolerance:
                    viable_models.append([model_instance.best_estimator_, l_window, score])

                if score < best_score:
                    print('we have a new best model!')
                    print('current score: {}'.format(score))
                    best_score = score
                    best_model = [model_instance.best_estimator_, l_window, score]

    if plot_comparison is False:
        pass
    else:  # if we do want to plot a comparison between real and predicted values,
        if test_data is None: # we get the X,y data using either test-train split or specified values
            y, X = get_Xy_2(m, lbounds=best_model[1], target_chem=target_chem)
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=tts_test_size, random_state=tts_random_state)
        else:
            y_train, X_train = get_Xy_2(m, lbounds=best_model[1], target_chem=target_chem)
            y_test, X_test = get_Xy_2(test_data, lbounds=best_model[1], target_chem=target_chem)

        y_pred = best_model[0].predict(X_test)
        metric_train = metric(y_train, best_model[0].predict(X_train))
        metric_test = metric(y_test, y_pred)

        plot_metric(y_test, y_train, y_pred, metric_label, metric_test, metric_train,
                    savefile=plot_comparison_savefile + metric_label, wl_window=best_model[1], display=True)
    return viable_models, best_model


def get_window_list(start: float, end: float, nwindows: list = None, width: float = None):
    if (nwindows is None) and (width is None):
        print('nwindows or width must be specified.')
        print('A default of 1 window has been applied')
        nwindows = 1

    # get nwindows if width is provided
    if nwindows is None:
        nwindows = int((end - start) / width)
    else:  # if nwindows is provided
        # check nwindows is int
        if type(nwindows) is int:
            pass
        else:
            print('nwindows is not an int.')

        width = int((end - start) / nwindows)

    starts = np.linspace(start, end - width, nwindows).reshape((-1, 1))
    ends = np.linspace(start + width, end, nwindows).reshape((-1, 1))

    windows = np.concatenate((starts, ends), axis=1)
    return windows


