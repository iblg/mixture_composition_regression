import sklearn as skl
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared

from sklearn.metrics import mean_squared_error

import numpy as np
from mixture_composition_regression.tests.import_training_set import import_training_set
from mixture_composition_regression.preprocessor_pipeline import *


# They need to be able to try different models, different params within those models, and different wavelength ranges

def cv_on_model_and_wavelength(m, wl, models, ycol=None, tts_test_size=None, tts_random_state=None, mae_tolerance=0.01):
    """

    :param mae_tolerance:
    :param tts_random_state:
    :param tts_test_size:
    :param ycol:
    :param m: Mixture
    :param wl: wavelength list
    :param models: GridSearchCV objects
    :return:
    """
    best_mae_test = 10 ** 10
    viable_models = []
    first = 0
    for idx, model in enumerate(models):
        print('\n \n \nModel number {}'.format(idx + 1))
        print(model.estimator)
        for l_window in wl:
            l_window[0] -= 1E-5  # this stops the bottom-most interval from being shorter than the others.
            y, X = get_Xy(m, lbounds=l_window, ycol=ycol)  # get y, X data

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts_test_size,
                                                                random_state=tts_random_state)
            model_instance = model.fit(X_train,
                                       y_train)  # model instance is the model with optimized params by gridsearch CV

            # Evaluate the model
            y_pred = model_instance.predict(X_test)
            # mae_test = mean_squared_error(y_train, model_instance.predict(X_train))
            # mae_train = mean_squared_error(y_test, y_pred)
            mae_train = median_absolute_error(y_train, model_instance.predict(X_train))
            mae_test = median_absolute_error(y_test, y_pred)

            if mae_test < mae_tolerance:
                viable_models.append([model_instance.best_estimator_, l_window, mae_test])

            if mae_test < best_mae_test:
                # print(l_window)
                # print(mae_test)
                # print(model_instance)
                # print(model_instance.best_params_)
                best_mae_test = mae_test
                best_model = [model_instance.best_estimator_, l_window, mae_test]
                # plot_mae(y_test, y_train, y_pred, mae_test, mae_train)

            # print('We\'ve converged on the params for this wavelength window.')
            # # print('l_window: {}'.format(l_window))
            # print(model_instance.score(X_test, y_test))
            # print(model_instance.best_params_)
    return viable_models, best_model


def get_window_list(start, end, nwindows=None, width=None):
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
    m, m2, m3 = import_training_set()  # create a mixture
    m.plot_by(idx=1, savefig='plotby', alpha=1, logy=True, cmap_name='viridis', spect_bounds=[1200, 3000])

    lbounds = [1200, 3200]  # set global bounds on wavelength
    nwindows = 30
    wl = get_window_list(lbounds[0], lbounds[1], nwindows=nwindows)  # get a list of windows you want to look at
    # best model so far: lbounds 2027, 2050, 10**-3 alpha, Ridge()
    sc = 'r2'
    ridge = GridSearchCV(
        Ridge(),
        # {'alpha': np.logspace(-10, 10, 11)}
        {'alpha': np.logspace(-5, 5, 11)},
        scoring=sc,
        cv=5
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
        KNeighborsRegressor(),
        {'n_neighbors': 5 + np.arange(10)},
        scoring=sc
    )

    cv_models = [
        ridge,
        kr,
        svr,
        knnr
    ]
    random_state = 42
    tts_size = 0.25
    # ycol = 0  # water
    ycol = 1  # dipa
    # ycol = 2  # salt
    viable_models, best_model = cv_on_model_and_wavelength(m, wl, cv_models,
                                                           ycol=ycol,
                                                           tts_test_size=tts_size,
                                                           tts_random_state=random_state,
                                                           mae_tolerance=1E-3)

    print('Best model:')
    print(best_model[1])
    print(best_model[0])
    #
    y, X = get_Xy(m, lbounds=best_model[1], ycol=ycol)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts_size,
                                                        random_state=random_state)

    y_pred = best_model[0].predict(X_test)
    mae_train = median_absolute_error(y_train, best_model[0].predict(X_train))

    mae_test = median_absolute_error(y_test, y_pred)
    plot_mae(y_test, y_train, y_pred, mae_test, mae_train, wl_window=best_model[1])
    return


if __name__ == '__main__':
    main()
