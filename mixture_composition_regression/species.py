import numpy as np
import pandas as pd
import xarray as xr

class Species:
    def __init__(self, name, properties = {}):
        '''
        Should return an xarray with coords of the dict.
        '''
        self.name = name
        self.prop_dict = properties
        da = xr.DataArray(name)
        for p_name, p_value in properties.items():
            ds = ds.assign_coords(dim=)


        self.da = da
        return

def main():
    nacl = Species('nacl', properties = {'mw':58.44})
    print(nacl.name)
    print(nacl.prop_dict)
    print(nacl.da)
    print(nacl.da.dims)

    return

if __name__ == '__main__':
    main()
