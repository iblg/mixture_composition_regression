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
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-22/2023-03-22.csv'
    df = clean_data(file)
    # print(df)
    cp = {'name':['nacl', 'water'] ,
          'mw':[58.44, 18.015],
          'nu': [2, 1]}
    s1 = Sample('s1', df, 0, 1, chem_properties = cp, w = [0.1, 0.9])
    s3 = Sample('s1', df, 4,5, chem_properties = cp, w = [0.1, 0.9])

    s2 = Sample('s2', df, 2, 3, chem_properties = cp, w = [0.2, 0.8])





    m1 = Mixture(s1, s3, s2)
    print(m1.da)
    # da = s1.da
    # da.loc[dict(nacl = 0.8, water = 0.2)] = s2.a

    # da3.loc[dict(x = 2)] = 100 ### this is how you modify by spot.
    return

if __name__ == '__main__':
    main()