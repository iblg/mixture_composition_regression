from mixture_composition_regression.examples.import_training_set import import_training_set
from mixture_composition_regression.cross_validation import cv_on_model_and_wavelength
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge


def main():
    water_dipa_nacl, water_dipa, water_nacl = import_training_set()

    lbounds = (900, 2000)  # set global bounds on wavelength

    mix_train = water_dipa + water_nacl
    mix_test = water_dipa_nacl.filter({'nacl': [10 ** -5, 1], 'dipa': [10 ** -5, 1]})

    nwindows = [1, 10]
    sc = 'neg_mean_absolute_error'
    random_state = 42
    tts_size = 0.25

    # metric = mean_absolute_percentage_error
    # metric_label = 'Mean abs fractional err'
    metric = mean_absolute_error
    metric_label = 'MAE'


    # m.plot_by(idx=2, savefig='plotby', alpha=1, logy=True, cmap_name='viridis', spect_bounds=lbounds, stylesheet=None)

    ridge = GridSearchCV(
        Ridge(), {'alpha': np.logspace(-7, 7, 14)}, scoring=sc, cv=5
    )

    kr_param_grid = {'kernel': ["rbf", 'linear'],
                    "alpha": np.logspace(-7, 7, 11),
                    "gamma": np.logspace(-7, 7, 11)}
    kr = GridSearchCV(
        KernelRidge(),
        param_grid=kr_param_grid,
        scoring=sc,
    )

    svr_param_grid = {'kernel': ['linear', 'rbf'],
         'gamma': ['scale', 'auto'],
         'epsilon': np.logspace(-7, 7, 10)
         }
    svr = GridSearchCV(
        SVR(),
        svr_param_grid,
        scoring=sc,
    )

    knnr_param_grid = {'n_neighbors': 5 + np.arange(5)}
    knnr = GridSearchCV(
        KNeighborsRegressor(), knnr_param_grid, scoring=sc
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

    viable_models, best_model = cv_on_model_and_wavelength(
        mix_train,
        nwindows,
        cv_models,
        target_chem='water',
        test_data=mix_test,
        tts_test_size=tts_size,
        tts_random_state=random_state,
        tolerance=5E-4,
        metric=metric,
        metric_label=metric_label,
        l_bounds=lbounds,
        plot_comparison=True,
        plot_comparison_savefile='../plots/axes_train'
    )



    # viable_models, best_model = cv_on_model_and_wavelength(water_dipa_nacl,
    #                                                        nwindows,
    #                                                        cv_models,
    #                                                        ycol=ycol,
    #                                                        tts_test_size=tts_size,
    #                                                        tts_random_state=random_state,
    #                                                        tolerance=5E-4,
    #                                                        metric=metric,
    #                                                        metric_label=metric_label,
    #                                                        l_bounds=lbounds,
    #                                                        plot_comparison=True,
    #                                                        plot_comparison_savefile='./plots/normal_tts'
    #                                                        )

    return


if __name__ == '__main__':
    main()
