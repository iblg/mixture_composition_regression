import numpy as np
import pandas as pd
import peak_resolver
from peak_resolver.linear_combination_fitting import lin_combination_fitting as lcf
import pandas as pd
import matplotlib.pyplot as plt


def main():
    directory = '/Users/ianbillinge/dev/mixture_composition_regression/mixture_composition_regression/examples/cellulose_example/data/'
    df = pd.read_csv(directory + 'all_spectra.csv')
    df = df.where(df['wavenumber'] > 500).dropna()
    basis = [df['s1t1'], df['s2t1'], df['s3t1']]

    target = df['s11t1']

    # diff = True
    diff = False

    if diff:
        basis = [i.diff() for i in basis]
        target = target.diff()
    res = lcf(target, basis, bounds=[[0, 1], [0, 1], [0, 1]])
    print(res)
    p = res['p']

    fig, ax = plt.subplots()
    x = df['wavenumber']
    for i in basis:
        ax.plot(x, i)
    ax.plot(x, target, color='#DE6FA1')

    ax.fill_between(x, np.zeros_like(x), target - p[0] * basis[0] - p[1] * basis[1] - p[2] * basis[2], color='#DE6FA1')
    # ax.fill_between(x, np.zeros_like(x), target - basis[2], color='#DE6FA1')

    plt.show()
    return


if __name__ == '__main__':
    main()
