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
from mixture_composition_regression.tests.import_training_set import import_training_set
from mixture_composition_regression.preprocessor_pipeline import *


# They need to be able to try different models, different params within those models, and different wavelength ranges

def cv_on_model_and_wavelength(m: mixture_composition_regression.mixture.Mixture,
                               nwindows: list,
                               models: list,
                               l_bounds: tuple,
                               ycol: int = None,
                               tts_test_size: float = None,
                               tts_random_state: int = None,
                               tolerance: float = 0.01,
                               metric: sklearn.metrics = None,
                               test_data: mixture_composition_regression.mixture.Mixture = None):

    if metric is None:
        metric = mean_squared_error
    else:
        pass

    best_metric = 10 ** 10

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
                    y, X = get_Xy(m, lbounds=l_window, ycol=ycol)  # get y, X data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts_test_size,
                                                                    random_state=tts_random_state)
                else:
                    y_train, X_train = get_Xy(m, lbounds=l_window, ycol=ycol)  # get y, X data
                    y_test, X_test = get_Xy(test_data, lbounds=l_window, ycol=ycol)


                model_instance = model.fit(X_train,
                                           y_train)  # model instance is the model with optimized params by gridsearch CV

                # Evaluate the model

                y_pred = model_instance.predict(X_test)
                train_eval = metric(y_train, model_instance.predict(X_train))
                test_eval = metric(y_test, y_pred)

                if test_eval < tolerance:
                    viable_models.append([model_instance.best_estimator_, l_window, test_eval])

                if test_eval < best_metric:
                    best_metric = test_eval
                    best_model = [model_instance.best_estimator_, l_window, test_eval]

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


def main():
    water_dipa_nacl, water_dipa, water_nacl = import_training_set()

    lbounds = [800, 3200]  # set global bounds on wavelength

    mix_test = water_dipa_nacl.filter({'nacl': [10 ** -5, 1], 'dipa': [10 ** -5, 1]})
    nwindows = [100]
    sc = 'neg_mean_squared_error'
    random_state = 42
    tts_size = 0.25
    ycol = 0  # water
    # ycol = 1  # dipa
    # ycol = 2  # salt
    # metric = mean_absolute_percentage_error
    # metric_label = 'Mean abs fractional err'
    metric = mean_absolute_error
    metric_label = 'MAE'

    mix_train = water_dipa + water_nacl

    # m.plot_by(idx=2, savefig='plotby', alpha=1, logy=True, cmap_name='viridis', spect_bounds=lbounds, stylesheet=None)

    ridge = GridSearchCV(
        Ridge(), {'alpha': np.logspace(-7, 7, 11)}, scoring=sc, cv=5
    )

    kr = GridSearchCV(
        KernelRidge(),
        param_grid={'kernel': ["rbf", 'linear'],
                    "alpha": np.logspace(-5, 5, 11),
                    "gamma": np.logspace(-5, 5, 11)},
        scoring=sc,
    )

    svr = GridSearchCV(
        SVR(),
        {'kernel': ['linear', 'rbf'],
         'gamma': ['scale', 'auto'],
         'epsilon': np.logspace(-5, 5, 10)
         },
        scoring=sc,
    )

    knnr = GridSearchCV(
        KNeighborsRegressor(), {'n_neighbors': 5 + np.arange(10)}, scoring=sc
    )

    mlp = GridSearchCV(
        MLPRegressor(solver='lbfgs', max_iter=400),
        {'hidden_layer_sizes': [10, 50, 100]},
        scoring=sc
    )

    cv_models = [
        ridge,
        # kr,
        # svr,
        # knnr,
        # mlp,
    ]

    viable_models, best_model = cv_on_model_and_wavelength(water_dipa, nwindows, cv_models, ycol=ycol,
                                                           tts_test_size=tts_size,
                                                           tts_random_state=random_state,
                                                           tolerance=5E-4,
                                                           metric=metric,
                                                           l_bounds=lbounds,
                                                           test_data=mix_test
                                                           )

    print('Best model:')
    print(best_model[1])
    print(best_model[0])

    # FOR IF YOU DON'T SUPPLY TEST_DATA
    # y, X = get_Xy(mix_train, lbounds=best_model[1], ycol=ycol)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts_size,
    #                                                     random_state=random_state)

    # FOR IF YOU DO SUPPLY TEST_DATA
    y_train, X_train = get_Xy(mix_train, lbounds=best_model[1], ycol=ycol)
    y_test, X_test = get_Xy(mix_test, lbounds=best_model[1], ycol=ycol)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts_size,
    #                                                     random_state=random_state)

    y_pred = best_model[0].predict(X_test)
    metric_train = metric(y_train, best_model[0].predict(X_train))
    metric_test = metric(y_test, y_pred)
    plt.style.use('default')
    plot_metric(y_test, y_train, y_pred, metric_label, metric_test, metric_train,
                savefile='./plots/nacl' + metric_label, wl_window=best_model[1], display=False)
    return


if __name__ == '__main__':
    main()
