from mixture_composition_regression.sample import Sample
from mixture_composition_regression.import_spectrum import clean_data
import pandas as pd


def main():
    cp = {'name': ['water', 'dipa', 'nacl'],
          'mw': [18.015, 101.19, 58.44],
          'nu': [1, 1, 2]}

    # 03-03-2023 data
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-03/2023-03-03.csv'
    df = clean_data(file)
    water1 = Sample('water1', df, 2, 3, chem_properties=cp, w=[1., 0., 0.])
    dipa1 = Sample('dipa1', df, 4, 5, chem_properties=cp, w=[0., 1., 0.])
    fig, ax = dipa1.plot(savefile='dipa', log_y=True)
    # water1.plot(savefile='water', log_y=True, fig=fig, ax=ax)
    water1.plot(savefile='water', log_y=False)
    print(pd.concat([water1.l, water1.a], axis='columns'))

    return


if __name__ == '__main__':
    main()
