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

    def __init__(self,
                 samples,
                 attrs=None,
                 savefile=None,
                 savefile_mode='w'):
        """
        Create a Mixture object.

        samples : list of mixture_composition_regression Sample objects
        these samples will form a training dataset for the machine learning model.

        attrs : dict-like.
        attributes of the mixture. Examples could include the names of chemicals, their molecular weights, etc.

        savefile_mode :
        """
        coordinates = [i for i in samples[0].da.coords][1:]  # get list of coordinates except for l.

        _check_samples(samples)  # check the samples for uniqueness etc.

        da_list = []
        for s in samples:
            da_list.append(s.da)

        # get attributes
        if attrs is None:
            # attrs = samples[0].attrs
            attrs = None
        else:
            pass

        da = xr.concat(da_list, dim='name')
        self.da = da
        self.chem_properties = samples[0].chem_properties

        # check savefile mode
        _check_savefile_mode(savefile_mode)

        if savefile is None:
            pass
        else:
            da.to_netcdf(savefile, mode=savefile_mode)

        return

    def __add__(self, other):
        _check_chem_properties(self, other)
        self.da = xr.concat([self.da, other.da], dim='name')
        return self


def _check_samples(samples):
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
    return


def _check_savefile_mode(savefile_mode):
    if savefile_mode == 'w':
        pass
    elif savefile_mode == 'a':
        pass
    else:
        print('savefile mode for mixture must be either \'w\' or \'a\' but is neither.')
    return


def _check_chem_properties(first, second):
    if first.chem_properties == second.chem_properties:
        pass
    else:
        print('First mixture and second mixture do not have the same chemical properties.')
    return