import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from import_spectrum import clean_data


class Sample:
    def __init__(self, name, data, l_col_idx, data_col_idx, chem_properties=None, w=None, savefile=None,
                 w_tol=10 ** (-4.), background=None):
        """
        Class for a single sample.



        Parameters
        ----------
        name : str
        The name of the sample.

        l_col_index : int
        The column index of the wavelength.

        data_col_idx : int
        The column index of the data (i.e., absorption, transmission).

        data : array-like
        The data (wavelength + absorption or transmission) associated with this sample. data must be array-like, with
        wavelengths stored in rows and the sample absorbance directly to

        background : NIR_sample
        The background spectrum to be subtracted.

        w : array-like
        The weight fractions of the different chemicals.

        w_tol : float, default 10**(-4).
        The error tolerance for the sum of weight fractions.

        chem_properties: dict
        Keys of this dict should be ID of components. Vals should be a dict, where keys are names of parameters and vals
        are values of those parameters. A simple example for NaCl/water mixture would be
        species_properties = {'nacl': {'mw': 58.44}, 'water': {'mw': 18.015}}

        savefile : str
        The filepath where you want to save to.

        Returns
        -------


        Examples
        --------

        """

        self.wa = None
        self.name = name

        self.l = data.iloc[:, l_col_idx]
        self.a = data.iloc[:, data_col_idx]
        # self.la = np.concat(self.l, self.a, axis = 'columns')
        self.chem_properties = chem_properties

        self.w_tol = w_tol
        self.check_chem_properties(chem_properties)  # ensure that w and chem_properties have same keys
        self.check_w(w, chem_properties)  # check whether the weights sum to 1.

        self.w = w

        dimensions = ['name', 'l']
        coordinates = {'l': self.l, 'name': [self.name]}
        da = xr.DataArray(self.a.values.reshape(1, -1), dims=dimensions, coords=coordinates)

        composition = []
        for idx, chem in enumerate(chem_properties['name']):
            composition.append(w[idx])
            da = da.assign_coords({chem: ("name", [w[idx]])})
        self.da = da

        if savefile:
            self.savefile = savefile
            self.save_to_file()
        return

    def plot(self, log_y=False, display=True, savefile=None):
        """
        Plots the data associated with the sample. Contains options for displaying and saving the plot.

        log_y : bool
        Default False. If True, scale of y axis will be y.

        display : bool
        Default True. If True, plot will be displayed. If False, plot will not be displayed.

        savefile : str
        Default None. If not None,

        """
        fig, ax = plt.subplots()
        ax.plot(self.l, self.a)
        ax.set_title(self.name)
        ax.set_xlabel(r"Wavelength, $\lambda$ [nm]")
        ax.set_ylabel(r"Absorption")

        if log_y == True:
            ax.set_yscale("log")

        if display:
            plt.show()

        if savefile is not None:
            fig.savefile(savefile)

        return

    def save_to_file(self):
        # ww = pd.DataFrame(self.ww * np.ones_like(self.a).reshape(-1,1))
        try:
            print("Sample: {}".format(self.name))
            ww = pd.DataFrame(self.ww * np.ones_like(np.array(self.l)), columns=["ww"])
            wa = pd.DataFrame(self.wa * np.ones_like(np.array(self.l)), columns=["wa"])

            to_file = pd.concat([self.l, self.a, ww, wa], axis="columns")
        except TypeError as te:
            print(te)
            to_file = pd.concat([self.l, self.a], axis="columns")
        # to_file = pd.concat([self.l, self.a],  axis = 'columns')
        to_file = to_file.rename(columns={to_file.columns[0]: "wavelength", to_file.columns[1]: "absorption"})

        to_file.to_csv(self.savefile + ".csv", index=False)
        return

    # def int_peak(self, low, high, log):
    #     """
    #     Integrates the peak defined by low and high wavelengths.
    #     """
    #     y = self.a.where((self.l < high) & (self.l > low)).dropna()
    #     x = self.l.where((self.l < high) & (self.l > low)).dropna()
    #
    #     if log == True:
    #         y = np.log10(y)
    #
    #     y0 = y.iloc[-1]
    #     y1 = y.iloc[0]
    #     x0 = x.iloc[-1]
    #     x1 = x.iloc[0]
    #     baseline = y0 + (y1 - y0) / (x1 - x0) * (x - x0)  # x0 is lowest wavelength
    #
    #     y = y - baseline
    #     area = np.trapz(y)
    #
    #     return area

    def check_w(self, w, chem_properties):

        # Check whether there are the same number of weights as there are chemicals.
        if len(w) == len(chem_properties['name']):
            pass
        else:
            print('w is a different length from chem_properties in sample {}. '
                  'Ensure that there are an equal number of weight fractions and chemicals'.format(self.name))

        # Check whether the weights add to 1.
        sum = 0
        for i in w:
            sum += i

        if np.abs(sum - 1) < self.w_tol:
            pass
        else:
            print('Weights do not sum to 1 in sample {}'.format(self.name))
        return

    def check_chem_properties(self, chem_properties: dict) -> None:
        ## future development: This should check that the keylist in w
        cp = chem_properties  # for notational ease
        nchems = len(cp['name'])

        for key, val in chem_properties.items():
            if len(val) == nchems:
                pass
            else:
                print('Property {} in chem_values is different length than number of chemicals!'.format(key))

        pass
        #
        # print(w.keys())
        # print(chem_properties['name'])
        # if np.array(w.keys()) == np.array(chem_properties['name']):
        #     print('They are the same')
        #     pass
        # else:
        #     print('keys of w and chem_properties are not the same for sample {}'.format(self.name))
        #
        # for kv1, kv2 in zip(w.items(),chem_properties.items()):
        #     print(kv1)
        #     print(kv2)
        # #Future development should include something that goes through each key in
        #
        # return


#
# def int_peak(samples, low, high):
#     areas = []
#     for sample in samples:
#         y = sample.a.where((sample.l < high) & (sample.l > low)).dropna()
#         x = sample.l.where((sample.l < high) & (sample.l > low)).dropna()
#
#         y0 = y.iloc[-1]
#         y1 = y.iloc[0]
#         x0 = x.iloc[-1]
#         x1 = x.iloc[0]
#         baseline = y0 + (y1 - y0) / (x1 - x0) * (x - x0)  # x0 is lowest wavelength
#
#         y = y - baseline
#         area = np.trapz(y)
#         areas.append(area)
#     return areas


def main():
    file = '/Users/ianbillinge/Documents/yiplab/programming/uvvisnir/1mm_pl/2023-03-22/2023-03-22.csv'
    df = clean_data(file)
    # print(df)
    cp = {'name': ['nacl', 'water'],
          'mw': [58.44, 18.015],
          'nu': [2, 1]}
    s1 = Sample('s1', df, 0, 1, chem_properties=cp, w=[0.1, 0.9])
    print(s1.da)

    return


if __name__ == "__main__":
    main()
