import numpy as np
import pandas as pd
import xarray as xr

class Species:
    def __init__(self, name, properties = {}):
        '''
        Should take in a list of samples and return a dataset where it has concatenated them along an axis.
        '''
        self.name = name
        self.prop_dict = properties
        da = xr.DataArray(name)
        # for p_name, p_value in properties.items():
        #     ds = ds.assign_coords(dim=)


        self.da = da
        return

def main():
    nacl = Species('nacl', properties = {'mw':58.44})


    return

if __name__ == '__main__':
    main()
