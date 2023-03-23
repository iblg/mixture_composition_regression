import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors


class Sample:
    def __init__(self, name, col_index, data, background=None, w1=None, w2=None, savefile=None):
        """
        Class for a single sample.

        Parameters
        ----------
        name : str
        The name of the sample.

        col_index : int

        data : array-like
        The data (wavelength + absorption or transmission) associated with this sample.

        background : NIR_sample
        The background spectrum to be subtracted.

        ww : float
        Weight fraction water

        wa : float
        Weight fraction amine

        savefile : str
        The filepath where you want to save to.

        Returns
        -------


        Examples
        --------

        """

        self.name = name
        self.col = col_index
        self.l = data.iloc[:, self.col]
        self.ww = ww
        self.wa = wa
        self.savefile = savefile

        if background:
            self.a = data.iloc[:, self.col + 1] - background.a
        else:
            self.a = data.iloc[:, self.col + 1]

        self.wa_areas = self.get_wa_areas()
        self.log_wa_areas = self.get_log_areas()

        try:
            self.ww_array = self.ww * np.ones_like(self.a)
            self.wa_array = self.wa * np.ones_like(self.a)
        except TypeError as te:
            print(te)
            print("No ww provided")

        self.bounds = {
            "interval1": [283, 380],
            "interval2": [930, 1030],
            "interval3": [1048, 1082],
            "interval4": [1083, 1133],
            "interval5": [1310, 1360],
            "interval6": [1592, 1745],
        }

        self.areas = {}

        for area, bounds in self.bounds.items():
            l1 = self.l.where((self.l > bounds[0]) & (self.l < bounds[1])).dropna()
            a1 = self.a.where((self.l > bounds[0]) & (self.l < bounds[1])).dropna()
            self.areas[area] = np.trapz(a1, x=l1)

        if savefile:
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



    def int_peak(self, low, high, log):
        """
        Integrates the peak defined by low and high wavelengths.
        """
        y = self.a.where((self.l < high) & (self.l > low)).dropna()
        x = self.l.where((self.l < high) & (self.l > low)).dropna()

        if log == True:
            y = np.log10(y)

        y0 = y.iloc[-1]
        y1 = y.iloc[0]
        x0 = x.iloc[-1]
        x1 = x.iloc[0]
        baseline = y0 + (y1 - y0) / (x1 - x0) * (x - x0)  # x0 is lowest wavelength

        y = y - baseline
        area = np.trapz(y)

        return area


def int_peak(samples, low, high):
    areas = []
    for sample in samples:
        y = sample.a.where((sample.l < high) & (sample.l > low)).dropna()
        x = sample.l.where((sample.l < high) & (sample.l > low)).dropna()

        y0 = y.iloc[-1]
        y1 = y.iloc[0]
        x0 = x.iloc[-1]
        x1 = x.iloc[0]
        baseline = y0 + (y1 - y0) / (x1 - x0) * (x - x0)  # x0 is lowest wavelength

        y = y - baseline
        area = np.trapz(y)
        areas.append(area)
    return areas


if __name__ == "__main__":
    main()
