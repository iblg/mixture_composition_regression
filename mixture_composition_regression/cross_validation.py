import sklearn as skl
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import ExpSineSquared

import numpy as np
from mixture_composition_regression.tests.import_training_set import import_training_set
from mixture_composition_regression.preprocessor_pipeline import *


# They need to be able to try different models, different params within those models, and different wavelength ranges

def cv_on_model_and_wavelength(m, wl, models, tts_test_size=None, tts_random_state=None):

    '''

    :param m: Mixture
    :param wl: wavelength list
    :param models: GridSearchCV objects
    :param tts: train-test split
    :return:
    '''

    for idx, model in enumerate(models):
        print(idx)
        for l_window in wl:
            y, X = get_Xy(m, lbounds=l_window)  # get y, X data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tts_test_size, random_state = 42)
            model.fit(X_train, y_train)
            print('l_window: {}'.format(l_window))
            print(model.best_estimator_)
    return


def get_window_list(start, end, nwindows=None, width=None):
    if (nwindows is None) and (width is None):
        print('nwindows or width must be specified.')
        print('A default of 1 window has been applied')
        nwindows = 1

    # get nwindows if width is provided
    if nwindows is None:
        nwindows = int((end - start) / width)
    else: # if nwindows is provided
        # check nwindows is int
        if type(nwindows) is int:
            pass
        else:
            print('nwindows is not an int.')

        width = int((end - start) / nwindows)

    starts = np.linspace(start, end-width, nwindows).reshape((-1, 1))
    ends = np.linspace(start + width, end, nwindows).reshape((-1, 1))

    windows = np.concatenate((starts, ends), axis=1)
    return windows



def main():
    m = import_training_set()
    lbounds = [900, 3200]
    wl = get_window_list(900, 3200, nwindows=30)

    ridge = GridSearchCV(Ridge(), {'alpha': np.logspace(-10, 10, 11)})
    # svr = GridSearchCV(SVR(), {'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto'], 'epsilon': np.logspace(-5,5,10)})


    cv_models = [ridge]
                 # svr]

    cv_on_model_and_wavelength(m, wl, cv_models)
    return


if __name__ == '__main__':
    main()
