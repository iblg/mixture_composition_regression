from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def plot_learning_curve(model,
                        y,
                        X,
                        train_sizes=np.arange(5, 21),
                        cv=5,
                        scoring='neg_mean_absolute_error',
                        savefile=None):
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv,
        scoring=scoring
    )
    # LearningCurveDisplay.from_estimator(
    #     model, X, y, train_sizes=train_sizes, cv=5)
    # plt.show()

    fig = plt.figure()
    gs = GridSpec(1, 1, left=0.15, bottom=0.15, top=0.98, right=0.98)
    ax = fig.add_subplot(gs[0])
    train_sizes = np.array(train_sizes)
    train_scores = np.array(train_scores)
    valid_scores = np.array(valid_scores)

    ax.errorbar(train_sizes, train_scores.mean(axis=1), yerr=train_scores.std(axis=1), label='Training scores')
    ax.errorbar(train_sizes, valid_scores.mean(axis=1), yerr=valid_scores.std(axis=1), label='Validation scores')
    ax.legend()
    ax.set_xlabel('Number of training data')
    ax.set_ylabel(scoring)

    plt.show()
    if savefile is None:
        pass
    else:
        plt.savefig(savefile + '.png', dpi=400)

    return


def plot_validation_curve(model,
                          y,
                          X,
                          param: dict,
                          log_x: bool = False,
                          log_y: bool = False,
                          scoring: str = 'neg_mean_absolute_error',
                          cv: int = 5,
                          savefile: str = None):
    for key, val in param.items():
        param_name = key
        param_range = val
    print(model)
    train_scores, valid_scores = validation_curve(model, X, y,
                                                  param_name=param_name, param_range=param_range, cv=cv,
                                                  scoring=scoring)

    fig = plt.figure()
    gs = GridSpec(1, 1, left=0.15, bottom=0.15, top=0.98, right=0.98)
    ax = fig.add_subplot(gs[0])
    train_scores = np.array(train_scores)
    valid_scores = np.array(valid_scores)
    if log_y:
        train_scores, valid_scores = np.abs(train_scores), np.abs(valid_scores) # This is some pretty dodgy work.
    ax.errorbar(param_range, train_scores.mean(axis=1), yerr=train_scores.std(axis=1), label='Training scores')
    ax.errorbar(param_range, valid_scores.mean(axis=1), yerr=valid_scores.std(axis=1), label='Validation scores')
    print(param_range)
    print(train_scores.mean(axis=1))
    print(valid_scores.mean(axis=1))
    ax.set_xlabel(param_name)
    ax.set_ylabel(scoring)

    if log_x:
        ax.set_xscale('log')

    if log_y:
        ax.set_yscale('log')
        ax.set_ylabel('Absolute val of ' + scoring)


    plt.legend()
    plt.show()

    if savefile is None:
        pass
    else:
        fig.savefig(savefile + param_name +'.png', dpi=400)

    return


def plot_validation_curve_over_param_grid(model,
                                          y,
                                          X,
                                          param_grid: dict,
                                          log_x: bool = False,
                                          log_y: bool = False,
                                          scoring: str = 'neg_mean_absolute_error',
                                          cv: int = 5,
                                          savefile: str = None):
    for key, val in param_grid.items():
        plot_validation_curve(model, y, X, param={key: val},
                              log_x=log_x, log_y=log_y, scoring=scoring, cv=cv, savefile=savefile)

    return
