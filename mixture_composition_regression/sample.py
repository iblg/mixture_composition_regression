import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mixture_composition_regression.import_spectrum import clean_data

class Sample:
    def __init__(self, name, data,
                 x_col_idx=None,
                 a_col_idx=None,
                 x_col_name=None,
                 a_col_name=None,
                 chem_properties=None,
                 w=None,
                 savefile=None,
                 w_tol=10 ** (-4.),

                 background=None):
        """
        Class for a single sample.



        Parameters
        ----------
        name : str
        The name of the sample.

        x_col_index : int
        The column index of the wavelength.

        a_col_idx : int
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

        self.x, self.a = self.get_x_a(data, x_col_idx, a_col_idx, x_col_name, a_col_name)

        # self.x = data.iloc[:, x_col_idx]
        # self.a = pd.DataFrame(data.iloc[:, np.c_[x_col_idx, data_col_idx]])
        # self.a = pd.DataFrame(data.iloc[:, data_col_idx])
        # self.a = self.a.set_index(self.a.iloc[:, 0]).sort_index()
        # self.la = np.concat(self.x, self.a, axis = 'columns')
        self.chem_properties = chem_properties

        self.w_tol = w_tol
        self.check_chem_properties(chem_properties)  # ensure that w and chem_properties have same keys
        self.check_w(w, chem_properties)  # check whether the weights sum to 1.

        self.w = self.get_w(w, chem_properties)
        # w = pd.Series(w, index=chem_properties['name']) #this has problems if it is passed a Pandas Series.
        # self.w = w

        dimensions = ['name', 'x']
        coordinates = {'x': self.x, 'name': [self.name]}
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

    def get_w(self, w, chem_properties):
        if isinstance(w, list):
            # print('w is a list')
            w = pd.Series(w, index = chem_properties['name'])
        elif isinstance(w, pd.Series):
            # print('w is a pd.Series')
            w = pd.DataFrame(w, index = chem_properties['name'])
        elif isinstance(w, pd.DataFrame):
            # print('w is a pd.DataFrame.')
            w = w.set_index(chem_properties['name'])
        return w

    def get_x_a(self, data, x_col_idx, a_col_idx, x_col_name, a_col_name):
        self.check_x_a(x_col_idx, a_col_idx, x_col_name, a_col_name)

        if a_col_name == 'sample_name': #for spreadsheets where the data column is the same as sample name
            a_col_name = self.name

        if x_col_name and a_col_name:
            x = data[x_col_name]

            xa = data[[x_col_name, a_col_name]]

        elif x_col_idx and a_col_idx:
            x = data.iloc[:, x_col_idx]

            a = data.iloc[:, a_col_idx]

            xa = pd.DataFrame([x, a], columns=['x', 'a'])
            print('xa: {}'.format(xa))
        else:
            print('Problems finding correct columns for sample {}'.format(self.name))
            return

        a = xa.set_index(xa[x_col_name]).sort_index()
        a = a.iloc[:, -1]

        return x, a

    def plot(self, log_y=False, display=True, savefile=None, fig=None, ax=None):
        """
        Plots the data associated with the sample. Contains options for displaying and saving the plot.

        log_y : bool
        Default False. If True, scale of y axis will be y.

        display : bool
        Default True. If True, plot will be displayed. If False, plot will not be displayed.

        savefile : str
        Default None. If not None,

        """
        if (fig is not None) & (ax is not None):
            fig, ax = fig, ax
        else:
            fig, ax = plt.subplots()
        ax.plot(self.x, self.a)
        ax.set_title(self.name)
        ax.set_xlabel(r"Wavelength, $\lambda$ [nm]")
        ax.set_ylabel(r"Absorbance")

        plt.tight_layout()

        if log_y == True:
            ax.set_yscale("log")

        if display:
            plt.show()

        if savefile is not None:
            fig.savefig(savefile + '.png', dpi=500)
            fig.savefig(savefile + '.pdf')

        return fig, ax

    def save_to_file(self):
        # ww = pd.DataFrame(self.ww * np.ones_like(self.a).reshape(-1,1))
        try:
            print("Sample: {}".format(self.name))
            ww = pd.DataFrame(self.ww * np.ones_like(np.array(self.x)), columns=["ww"])
            wa = pd.DataFrame(self.wa * np.ones_like(np.array(self.x)), columns=["wa"])

            to_file = pd.concat([self.x, self.a, ww, wa], axis="columns")
        except TypeError as te:
            print(te)
            to_file = pd.concat([self.x, self.a], axis="columns")
        # to_file = pd.concat([self.x, self.a],  axis = 'columns')
        to_file = to_file.rename(columns={to_file.columns[0]: "wavelength", to_file.columns[1]: "absorption"})

        to_file.to_csv(self.savefile + ".csv", index=False)
        return

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

    def check_chem_properties(self, chem_properties: dict):
        ## future development: This should check that the keylist in w
        cp = chem_properties  # for notational ease
        nchems = len(cp['name'])

        for key, val in chem_properties.items():
            if len(val) == nchems:
                pass
            else:
                print('Property {} in chem_values is different length than number of chemicals!'.format(key))

        return
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

    def check_x_a(self, x_col_idx, a_col_idx, x_col_name, a_col_name):
        # print('Sample: {}'.format(self.name))
        # print('x_col_idx: {}'.format(x_col_idx))
        # print('x_col_name: {}'.format(x_col_name))
        # print('a_col_idx: {}'.format(a_col_idx))
        # print('a_col_name: {}'.format(a_col_name))

        if (x_col_idx is not None) and (a_col_idx is not None):
            return
        if (x_col_name is not None) and (a_col_name is not None):
            return
        else:
            print('There was a problem with the data in sample {}.'.format(self.name))
            print('Either a x_col_idx AND an a_col_index or a x_col_name AND an a_col_name must be defined.')
            return


def main():
    # file = '/Users/ianbillinge/Documents/yiplab/projects/ir/cellulose/all_spectra.csv'
    # df = clean_data(file)
    #
    #
    #
    # w_file = '/Users/ianbillinge/Documents/yiplab/projects/ir/cellulose/composition.csv'
    # composition = pd.read_csv(w_file)
    # samples = np.array(composition.columns)[1:]
    #
    # cp = {'name': ['cellulose', 'hemicellulose', 'lignin'],
    #       'mw': [1, 1, 1],
    #       'nu': [1, 1, 1]}
    # ds = []
    # for s in samples:
    #     ds.append(Sample(s, df, x_col_name='wavenumber', a_col_name=s, chem_properties=cp, w=composition[s]/100.))

    return


if __name__ == "__main__":
    main()
