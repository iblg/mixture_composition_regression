import numpy as np
import xarray as xr
import mixture_composition_regression
from mixture_composition_regression.import_spectrum import clean_data
from mixture_composition_regression.sample import Sample
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from itertools import count


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
                 name=None,
                 sort_da_by_l=True):
        """
        Create a Mixture object.

        samples : list of mixture_composition_regression Sample objects
        these samples will form a training dataset for the machine learning model.

        attrs : dict-like.
        attributes of the mixture. Examples could include the names of chemicals, their molecular weights, etc.

        savefile_mode :
        """
        self.name = name
        coordinates = [i for i in samples[0].da.coords][1:]  # get list of coordinates except for l.
        print('Checking samples in mixture {}'.format(self.name))
        samples = _check_samples(samples)  # check the samples for uniqueness etc.

        da_list = []
        for s in samples:
            da_list.append(s.da)

        # get attributes
        if attrs is None:
            # attrs = samples[0].attrs
            self.attrs = None
        else:
            self.attrs = attrs
            pass

        da = xr.concat(da_list, dim='name')
        if sort_da_by_l:
            self.da = da.sortby(['l'])
        else:
            self.da = da
        self.chem_properties = samples[0].chem_properties
        self.samples = samples
        return

    def __add__(self, other):
        _check_chem_properties(self, other)
        # self.da = xr.concat([self.da, other.da], dim='name')
        # self.da = self.da.sortby(['l'], ascending=True)
        [self.samples.append(s) for s in other.samples]
        mix = Mixture(self.samples, attrs=self.attrs)
        return mix

    def set_name(self, new_name):
        new_mix = Mixture(self.samples, attrs=self.attrs, name=new_name)
        return new_mix

    def savefile(self, savefile, mode='w'):
        _check_savefile_mode(mode)

        self.da.to_netcdf(savefile, mode=mode)
        return

    def plot_by(self, idx=0,
                cmap_name='cividis',
                savefig=None,
                alpha=1,
                logy=False,
                spect_bounds=None,
                xlabel='Wavelength [nm]',
                ylabel='Absorption [–]',
                stylesheet=None
                ):
        if stylesheet is None:
            pass
        else:
            plt.style.use(stylesheet)
        fig = plt.figure()
        gs = GridSpec(1, 1, left=0.20, bottom=0.20, right=0.95, top=0.95)
        ax = fig.add_subplot(gs[0])

        cmap = cm.get_cmap(cmap_name)
        # x = []
        # for s in self.samples:
        #     x.append(s.w[idx])
        x = [s.w[idx] for s in self.samples]

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(x), vmax=max(x)))
        plt.colorbar(sm, ax=ax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        for s in self.samples:
            c = s.w[idx]
            if spect_bounds is None:
                l, a = s.l, s.a
            else:
                l = s.l.where((s.l > spect_bounds[0]))
                a = s.a.where((s.l > spect_bounds[0]))
                l = l.where((s.l < spect_bounds[1]))
                a = a.where((s.l < spect_bounds[1]))
            ax.plot(l, a, color=cmap(c), alpha=alpha)

        # plt.colorbar(ax, cax=cbar)
        if logy is True:
            ax.set_yscale('log')

        if savefig is None:
            pass
        else:
            plt.savefig(savefig + '.png', dpi=400)

        return

    def filter(self, criteria):

        # check crits
        if isinstance(criteria, dict):
            pass
        else:
            print('One criterion was passed to Mixture.filter() that is not a dict.')

        m = []

        for sample in self.samples:
            include = True
            for chem_name, bds in criteria.items():
                if bds[0] <= sample.w[chem_name] <= bds[1]:
                    pass
                else:
                    include = False
            if include:
                m.append(sample)
        m = Mixture(m, attrs=self.attrs)

        return m


def _check_samples(samples):
    """
    Checks whether all items in a list of samples are Sample objects.
    Then removes all duplicate Samples.
    :param samples: a list of Sample objects.
    samples should be a mixture_composition_regression.mixture.Mixture.samples
    :return:
    """
    # check that all are samples
    for s in samples:
        if isinstance(s, mixture_composition_regression.sample.Sample):
            pass
        else:
            print('Mixture __init__ got passed something that isn\'t a sample!')

    # initialize a null list
    unique_samples = []
    # traverse for all elements
    for s in samples:
        if s not in unique_samples:
            unique_samples.append(s)
        else:
            print('Sample {} has a duplicate!'.format(s.name))

    return unique_samples


def _check_savefile_mode(savefile_mode):
    """
    This ensures that savefile_mode is being passed an appropriate mode (either to write or append).
    :param savefile_mode: str. Should be 'w' for write or 'a' for append.
    :return:
    """
    if savefile_mode == 'w':
        pass
    elif savefile_mode == 'a':
        pass
    else:
        print('savefile mode for mixture must be either \'w\' or \'a\' but is neither.')
    return


def _check_chem_properties(first: Mixture, second: Mixture):
    """
    This function checks that the chemical properties of two mixtures are identical when they are being added.
    :param first: mixture_composition_regression.mixture.Mixture
    The first mixture being added.
    :param second: mixture_composition_regression.mixture.Mixture
    The second mixture being added.
    :return:
    """
    if first.chem_properties == second.chem_properties:
        pass
    else:
        print('First mixture {} and second mixture {} do not have the same chemical properties.'.format(first.name,
                                                                                                        second.name))
    return
