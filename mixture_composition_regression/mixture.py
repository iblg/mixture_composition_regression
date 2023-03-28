import numpy as np
import xarray as xr
import mixture_composition_regression
from mixture_composition_regression.import_spectrum import clean_data
from mixture_composition_regression.sample import Sample


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
            if isinstance(s, mixture_composition_regression.sample.Sample):
                pass
            else:
                print('Mixture __init__ got passed something that isn\'t a sample!')

        # check for duplicate samples. If duplicates, then add a trivial amount onto each coord.
        for s in range(len(samples)):
            for s2 in range(s + 1, len(samples)):
                if samples[s].name == samples[s2].name:
                    print('Two samples have duplicate names! {} and {}.'.format(samples[s].name, samples[s2].name))
        # for s in range(len(samples)):
        #     for s2 in range(s, len(samples)):
        #         # print('Comparing samples {} and {}'.format(samples[s].name, samples[s2].name))
        #         try:
        #             # try to have them join exactly.
        #             # if this succeeds, they are identical in composition, and we will proceed to add randomness.
        #             xr.align(samples[s].da, samples[s2].da, join='exact')
        #             # print(samples[s2].da.coords)
        #             for idx, coord in enumerate(coordinates):
        #                 # print(samples[s2].da.coords[coordinates[idx]])
        #                 samples[s2].da.coords[coordinates[idx]] = samples[s2].da.coords[
        #                                                               coordinates[idx]] + np.random.rand() * 10 ** -9
        #                 print('We added some randomness to sample {}'.format(samples[s2].name))
        #                 # print(samples[s2].da.coords[coordinates[idx]])
        #             print(samples[s2].da.coords)
        #
        #
        #         except ValueError:
        #             # If the two do not join, they do not have identical coordinates.
        #             # print('No need for randomness')
        #             pass

        da_list = []
        for s in samples:
            da_list.append(s.da)
        da = xr.concat(da_list, dim = 'name')
        # da = xr.combine_by_coords(da_list)  # this is the meaty part of the function. This does all the work.
        self.da = da
        self.chem_properties = samples[0].chem_properties
        return