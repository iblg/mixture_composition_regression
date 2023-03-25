import numpy as np
import pandas as pd
import xarray as xr
from import_spectrum import clean_data
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
        coordinates = [i for i in samples[0].da.coords][1:] #get list of coordinates except for l.

        #check that all are samples
        for s in samples:
            if isinstance(s, Sample):
                pass
            else:
                print('Mixture __init__ got passed something that isn\'t a sample!')

        #check for duplicate samples. If duplicates, then add a trivial amount onto one coord.
        for s in range(len(samples)):
            for s2 in range(s + 1, len(samples)):
                print('Comparing samples {} and {}'.format(samples[s].name, samples[s2].name))
                try:
                    xr.align(samples[s].da, samples[s2].da, join = 'exact')
                    samples[s2].da.coords[coordinates[-1]] =  samples[s2].da.coords[coordinates[-1]] + np.random.rand() * 10**-9
                    print('We added some randomness')
                except ValueError:
                    print('No need for randomness')
                    pass

        da_list = []
        for s in samples:
            da_list.append(s.da)
        da = xr.combine_by_coords(da_list) #this is the meatty part of the function. This does all the work.

        self.da = da
        return



def main():
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-22/2023-03-22.csv'
    df = clean_data(file)
    cp = {'name':['nacl', 'water'] ,
          'mw':[58.44, 18.015],
          'nu': [2, 1]}
    s1 = Sample('s1', df, 0, 1, chem_properties = cp, w = [0.1, 0.9])
    s2 = Sample('s2', df, 2, 3, chem_properties = cp, w = [0.2, 0.8])
    s3 = Sample('s3', df, 4, 5, chem_properties = cp, w = [0.2, 0.8])
    s4 = Sample('s4', df, 4, 5, chem_properties = cp, w = [0.2, 0.8])




    m1 = Mixture([s1, s2, s3, s4])
    print(m1.da)

    return

if __name__ == '__main__':
    main()