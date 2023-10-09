from mixture_composition_regression.preprocessor_pipeline import *
from mixture_composition_regression.examples.import_training_set import import_training_set
from sklearn.model_selection import train_test_split


def main():
    preprocessor = get_preprocessor()
    model = get_pipeline(preprocessor, regr=Ridge(alpha=10**-3))
    # model = get_pipeline(preprocessor, regr=RidgeCV(alphas=np.logspace(-20, 20, 21)))
    # model = get_pipeline(preprocessor, regr=SVR(gamma='scale', epsilon=0.001))
    # does well regressing DIPA between 2000, 2500
    m1 = import_training_set()
    lbounds = [2027, 2050]
    y, X = get_Xy(m1, lbounds=lbounds)
    # X = np.log10(X)
    y = y[:, 0]  # only regress water
    # y = y[:, 1]  # only regress dipa
    # y = y[:, 2]  # only regress nacl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_train = median_absolute_error(y_train, model.predict(X_train))
    mae_test = median_absolute_error(y_test, y_pred)
    plot_metric(y_test, y_train, y_pred, mae_test, mae_train, )

    return


if __name__ == '__main__':
    main()
