import sklearn
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import mixture_composition_regression.mixture

from mixture_composition_regression.preprocessor_pipeline import get_Xy
from mixture_composition_regression.preprocessor_pipeline import plot_metric

import numpy as np


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
    """

    :param m: mixture_composition_regression.mixture.Mixture object
    The mixture used to train and test the dataset.

    :param nwindows: list
    List of ints. The number of wavelength (or frequency) subdivisions to use when dividing up training spectra in this
    cross-validation search.

    :param models: list
    List of models to use in cross validation search.

    :param l_bounds: list
    The wavelength (or frequency) bounds on the total data.

    :param target_chem:

    :param tts_test_size:

    :param tts_random_state:

    :param tolerance:

    :param metric: sklearn.metrics function
    Metric used for goodness of fit.
    List of metrics can be found at scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

    :param metric_label: str
    Text label used for printing results of the goodness of fit metric.

    :param test_data:

    :param plot_comparison: bool, default False
    If False, no plot is made comparing

    :param plot_comparison_savefile:
    :return:
    """
    for x in nwindows:  # check nwindows
        if x is int:
            pass
        else:
            print('Number of window subdividions in cv_on_model_and_wavelength is not an integer value!')
            print('Please ensure that nwindows is a list of ints.')
            return

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
        if test_data is None:  # we get the X,y data using either test-train split or specified values
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
                    filepath=plot_comparison_savefile + metric_label, wl_window=best_model[1], display=True)

    best_y, best_X = get_Xy(m, lbounds=best_model[1], ycol=ycol)
    return viable_models, best_model, best_y, best_X


def get_window_list(start: float,
                    end: float,
                    nwindows: list = None,
                    width: float = None,
                    print_windows: bool = False):
    """
    Returns a list of wavelength (or frequency) intervals that demarcate fitting intervals for the model to train on.

    :param start: float
    The minimum wavelength (or frequency) considered by the model.

    :param end: float
    The maximum wavelength (or frequency) considered by the model.

    :param nwindows: int
    The number of windows to divide the wavelength (or frequency) interval into.
    Note: only nwindows OR width should be provided. Not both.

    :param width: float
    The width of each wavelength (or frequency) interval.
    Note: only nwindows OR width should be provided. Not both.
    
    :param print_windows: bool, default False
    If print_windows is True, the list of windows generated will get printed.

    :return:
    """
    if nwindows:  # if nwindows is passed
        if type(nwindows) is int:  # check that nwindows is correct type
            pass
        else:
            print('nwindows is not an int. get_window_list will fail.')
            return

    if (nwindows is None) and (width is None):  # if neither nwindows nor width is provided
        print('nwindows or width must be specified.')
        print('A default of 1 window has been applied')
        nwindows = 1
    elif (nwindows is None) and (width is not None):  # get nwindows if width is provided
        nwindows = int((end - start) / width)
    elif (nwindows is not None) and width is None:  # if nwindows is not none
        width = int((end - start) / nwindows)
    elif nwindows and width:
        print('Both nwindows and width provided.')
        print('get_window_list will revert to nwindows and ignore width.')


    starts = np.linspace(start, end - width, nwindows).reshape((-1, 1))
    ends = np.linspace(start + width, end, nwindows).reshape((-1, 1))

    windows = np.concatenate((starts, ends), axis=1)

    if print_windows:
        print(windows)

    return windows
