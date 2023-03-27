import numpy as np
import xarray as xr
from sample import Sample


class Mixture:
    """
    A container for your test (or test/train) data for a system of interest. Individual samples in this dataset can be
    loaded as samples.
    Contains an xarray DataArray with dims 'l' for wavelength and N different dims corresponding to the different
    chemicals in the mixture.

    samples : list of mixture_composition_regression.Sample objects
    samples is the full list of samples that you wish to include in your training or test/train dataset. All *samples
    must have the same dims. Handling of duplicates, i.e. ones with the same composition (and therefore same coords) is
    done by adding a random number scaled by 10**(-9).

    """

    def __init__(self, samples):
        coordinates = [i for i in samples[0].da.coords][1:]  # get list of coordinates except for l.

        # check that all are samples
        for s in samples:
            if isinstance(s, Sample):
                pass
            else:
                print('Mixture __init__ got passed something that isn\'t a sample!')

        # check for duplicate samples. If duplicates, then add a trivial amount onto one coord.
        for s in range(len(samples)):
            for s2 in range(s + 1, len(samples)):
                # print('Comparing samples {} and {}'.format(samples[s].name, samples[s2].name))
                try:
                    xr.align(samples[s].da, samples[s2].da, join='exact')
                    samples[s2].da.coords[coordinates[-1]] = samples[s2].da.coords[
                                                                 coordinates[-1]] + np.random.rand() * 10 ** -9
                    # print('We added some randomness')
                except ValueError:
                    # print('No need for randomness')
                    pass

        da_list = []
        for s in samples:
            da_list.append(s.da)
        da = xr.combine_by_coords(da_list)  # this is the meatty part of the function. This does all the work.

        self.da = da
        return


def main():
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
    print(m1.da)

    return


if __name__ == '__main__':
    main()
