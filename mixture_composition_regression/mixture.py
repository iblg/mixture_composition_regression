import numpy as np
import pandas as pd
import xarray as xr
from import_spectrum import clean_data
from sample import Sample

class Mixture:
    def __init__(self, *samples):
        coordinates = [i for i in samples[0].da.coords][1:] #get list of coordinates except for l.

        #check for duplicate samples. If duplicates, then add a trivial amount onto one coord.
        for s in samples[1:]:
            try:
                xr.align(s.da, samples[0].da, join='exact')
                s.da.coords[coordinates[-1]] = s.da.coords[coordinates[-1]] + 10**-9
            except ValueError:
                pass

        da_list = []

        for s in samples:
            da_list.append(s.da)

        da = xr.combine_by_coords(da_list)
        self.da = da
        return



def main():

    return

if __name__ == '__main__':
    main()