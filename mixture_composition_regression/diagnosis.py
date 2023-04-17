import mixture_composition_regression as mcr
from sklearn.model_selection import learning_curve
from mixture_composition_regression.cross_validation import *
from mixture_composition_regression.tests.import_training_set import import_training_set

def plot_learning_curve(model, train_sizes, cv=5):

    pass
    # return


def main():
    wdn, wd, wn = import_training_set()

    lbounds = [900, 3200]
    nwindows = [1]
    sc = 'r2'
    tts_size = 0.25
    ycol = 0  # water
    # ycol = 1  # dipa
    # ycol = 2  # salt
    metric = mean_absolute_percentage_error

    ridge = GridSearchCV(
        Ridge(),
        # {'alpha': np.logspace(-10, 10, 11)}
        {'alpha': np.logspace(-5, 5, 11)},
        scoring=sc,
        cv=5
    )

    cv_models = [ridge]
    random_state = 42

    viable_models, best_model = cv_on_model_and_wavelength(wdn, nwindows, cv_models, ycol=ycol, tts_test_size=tts_size,
                                                           tts_random_state=random_state, tolerance=5E-4,
                                                           metric=metric)
    # plot_learning_curve(best_model)

    return


if __name__ == '__main__':
    main()