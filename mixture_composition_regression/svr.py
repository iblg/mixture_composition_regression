# from typing import Dict
from sklearn.svm import SVR

import mixture_composition_regression.mixture
from mixture_composition_regression.sample import Sample
from mixture_composition_regression.mixture import Mixture
from mixture_composition_regression.import_spectrum import clean_data
import matplotlib.pyplot as plt


def import_training_set():
    cp = {'name': ['water', 'dipa', 'nacl'],
          'mw': [18.015, 101.19, 58.44],
          'nu': [1, 1, 2]}

    # 03-03-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-03/2023-03-03.csv'
    df = clean_data(file)
    water1 = Sample('water1', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])
    dipa1 = Sample('dipa1', df, 4, 5, chem_properties=cp, w=[0., 1., 0.])

    # 03-07-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-07/2023-03-07.csv'
    df = clean_data(file)
    water2 = Sample('water2', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])
    dipa2 = Sample('dipa2', df, 4, 5, chem_properties=cp, w=[0., 1., 0.])
    dipa_w1 = Sample('dipa_w1', df, 6, 7, chem_properties=cp,
                     w=[0.0910 / (0.0910 + 0.9474), 0.9474 / (0.0910 + 0.9474), 0.])
    dipa_w2 = Sample('dipa_w2', df, 8, 9, chem_properties=cp,
                     w=[0.1510 / (0.1510 + 1.0358), 1.0358 / (0.1510 + 1.0358), 0.])

    # 03-09-2023
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-09/2023-03-09.csv'
    df = clean_data(file)
    dipa_w1a = Sample('dipa_w1a', df, 0, 1, chem_properties=cp,
                      w=[0.0910 / (0.0910 + 0.9474), 0.9474 / (0.0910 + 0.9474), 0.])
    dipa_w2a = Sample('dipa_w2a', df, 2, 3, chem_properties=cp,
                      w=[0.1510 / (0.1510 + 1.0358), 1.0358 / (0.1510 + 1.0358), 0.])
    dipa_w3 = Sample('dipa_w3', df, 4, 5, chem_properties=cp,
                     w=[0.0382 / (0.0382 + 0.8671), 0.8671 / (0.0382 + 0.8671), 0.])
    dipa_w4 = Sample('dipa_w4', df, 6, 7, chem_properties=cp,
                     w=[0.3690 / (0.3690 + 1.1550), 1.1550 / (0.3690 + 1.1550), 0.])

    # 03-22-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-22/2023-03-22.csv'
    df = clean_data(file)
    water3 = Sample('water3', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])

    five_M = Sample('5M', df, 4, 5, chem_properties=cp, w=[1. - 0.2470, 0., 0.2470])
    five_M_2 = Sample('5M_2', df, 6, 7, chem_properties=cp, w=[1. - 0.2470, 0., 0.2470])
    two_M = Sample('2M', df, 8, 9, chem_properties=cp, w=[1. - 0.1087, 0., 0.1087])
    two_M_2 = Sample('2M_2', df, 10, 11, chem_properties=cp, w=[1. - 0.1087, 0., 0.1087])
    four_M = Sample('4M', df, 12, 13, chem_properties=cp, w=[1. - 0.2036, 0., 0.2036])
    four_M_2 = Sample('4M_2', df, 14, 15, chem_properties=cp, w=[1. - 0.2036, 0., 0.2036])

    m1 = Mixture([water1, dipa1, water2, water3, dipa2, dipa_w1, dipa_w1a, dipa_w2, dipa_w2a, dipa_w3, dipa_w4,
                  five_M, five_M_2, two_M, two_M_2, four_M, four_M_2])
    return m1

def import_small_training_set():
    cp = {'name': ['water', 'dipa', 'nacl'],
          'mw': [18.015, 101.19, 58.44],
          'nu': [1, 1, 2]}

    # 03-03-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-03/2023-03-03.csv'
    df = clean_data(file)
    water1 = Sample('water1', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])
    dipa1 = Sample('dipa1', df, 4, 5, chem_properties=cp, w=[0., 1., 0.])

    # 03-07-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-07/2023-03-07.csv'
    df = clean_data(file)
    water2 = Sample('water2', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])
    dipa2 = Sample('dipa2', df, 4, 5, chem_properties=cp, w=[0., 1., 0.])
    dipa_w1 = Sample('dipa_w1', df, 6, 7, chem_properties=cp,
                     w=[0.0910 / (0.0910 + 0.9474), 0.9474 / (0.0910 + 0.9474), 0.])
    dipa_w2 = Sample('dipa_w2', df, 8, 9, chem_properties=cp,
                     w=[0.1510 / (0.1510 + 1.0358), 1.0358 / (0.1510 + 1.0358), 0.])

    # 03-09-2023
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-09/2023-03-09.csv'
    df = clean_data(file)
    dipa_w1a = Sample('dipa_w1a', df, 0, 1, chem_properties=cp,
                      w=[0.0910 / (0.0910 + 0.9474), 0.9474 / (0.0910 + 0.9474), 0.])
    dipa_w2a = Sample('dipa_w2a', df, 2, 3, chem_properties=cp,
                      w=[0.1510 / (0.1510 + 1.0358), 1.0358 / (0.1510 + 1.0358), 0.])
    dipa_w3 = Sample('dipa_w3', df, 4, 5, chem_properties=cp,
                     w=[0.0382 / (0.0382 + 0.8671), 0.8671 / (0.0382 + 0.8671), 0.])
    dipa_w4 = Sample('dipa_w4', df, 6, 7, chem_properties=cp,
                     w=[0.3690 / (0.3690 + 1.1550), 1.1550 / (0.3690 + 1.1550), 0.])

    # 03-22-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-22/2023-03-22.csv'
    df = clean_data(file)
    water3 = Sample('water3', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])

    five_M = Sample('5M', df, 4, 5, chem_properties=cp, w=[1. - 0.2470, 0., 0.2470])
    five_M_2 = Sample('5M_2', df, 6, 7, chem_properties=cp, w=[1. - 0.2470, 0., 0.2470])
    two_M = Sample('2M', df, 8, 9, chem_properties=cp, w=[1. - 0.1087, 0., 0.1087])
    two_M_2 = Sample('2M_2', df, 10, 11, chem_properties=cp, w=[1. - 0.1087, 0., 0.1087])
    four_M = Sample('4M', df, 12, 13, chem_properties=cp, w=[1. - 0.2036, 0., 0.2036])
    four_M_2 = Sample('4M_2', df, 14, 15, chem_properties=cp, w=[1. - 0.2036, 0., 0.2036])

    m1 = Mixture([water1, water2, water3])
    return m1


def plot_single(m):
    fig, ax = plt.subplots()
    comp = {'water': 1, 'dipa': 0, 'nacl': 0}
    data = m.da.sel(comp, method='nearest')
    print('data:')
    print(data)
    ax.plot(data.l, data.values, '.')
    plt.show()
    return


def get_target_input(m: mixture_composition_regression.mixture.Mixture) -> object:
    """
    This function will take a mixture object, convert it to an appropriate format for
    sklearn, and do a test-train split.

    :param m: mixture_composition_regression.mixture.Mixture
    m should be a Mixture object.

    :return:
    """
    da = m.da
    coords = da.coords
    print(da)
    # del coords['l']
    chems = m.chem_properties['name']
    comps = []

    for idx, i in enumerate(da.coords[chems[0]]):
        component = i.name
        # print(i)
        # print(i.name)
        ccoords = {}
        for chem in chems:
            ccoords[chem] = da.coords[chem]
            # print(ccoords[chem].values[idx])



    return


def main():
    m1 = import_small_training_set()
    # plot_single(m1)
    get_target_input(m1)

    return


if __name__ == '__main__':
    main()
