import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def read_range_files(filenames: list):
    ranges = []

    for f in filenames:
        with open(f, 'r') as file:
            ranges.append(file.readlines())

    ranges = [float(i.strip()) for row in ranges for i in row]
    ranges = [[ranges[idx], ranges[idx + 1]] for idx, i in enumerate(ranges) if idx % 2 == 0]

    return ranges


def read_predictor_files(filenames: list):
    predictors = []
    for f in filenames:
        with open(f, 'rb') as file:
            predictors.append(pickle.load(file))
    return predictors


def read_uncertainty_files(filenames: list):
    u = [read_uncertainty_file(f) for f in filenames]
    return u


def read_uncertainty_file(f: str):
    """

    :param f: Filename
    :return:
    """
    with open(f, 'r') as file:
        data = file.readlines()

    data = [float(i.strip()) for i in data]
    return data


def predict_on_test_csvs(fpath, bestmodel_container, regressand, target, sample_name=None, print_sample=False,
                         printres=False, xgrid=None):
    new_data = pd.read_csv(fpath,
                           #                        names=['wavenumber', 'absorbance'],
                           header=0,
                           # it was reading in the first row as data and causing problems. So I just had it read the
                           # column names from the first row
                           dtype='float')

    # renamed for less typing, but you can absolutely get rid of these column names and just rename to your preference
    new_data = new_data.rename(columns={new_data.columns[0]: 'x', new_data.columns[1]: 'y'})

    if xgrid is None:  # if the data needs to be re-gridded
        pass
    else:
        new_data = regrid_ir_spectrum(new_data, xgrid)
    # because if you look higher in the code, we are currently regressing on the derivative of the data
    # so I calculated the derivative here
    if regressand == 'a':
        new_data[regressand] = new_data['y']
    if regressand == 'da':
        new_data[regressand] = new_data['y'].diff()
    elif regressand == 'd2a':
        new_data[regressand] = new_data['y'].diff(order=2)

    # get the wavelength window we care about and slice the data, only keeping that stuff
    window = bestmodel_container[1]
    new_data = new_data.where(new_data['x'] > window[0]).where(new_data['x'] < window[1]).dropna()
    new_data_dy = np.array(new_data[regressand]).reshape(1, -1)

    predictor = bestmodel_container[0]
    # try:
    prediction = predictor.predict(new_data_dy)
    # except ValueError as ve:
    #     print(ve)
    #     print('Sample: {}'.format(sample_name))
    #     print(new_data_dy)

    # discard unwanted nested lists
    for i in prediction.shape:
        prediction = prediction[0]

    if print_sample is True:
        print('Sample: {}'.format(sample_name))

    if printres:
        print("predicted composition {} {:1.3f}".format(target, prediction))
    return prediction


def regrid_ir_spectrum(data: pd.DataFrame, xgrid: np.array):
    # assuming the data has only absorption data in a column called 'y' and wavenumbers in a column called 'x'
    # do the regridding and add the new x axis as a column
    data2 = pd.DataFrame()
    # print(data)
    data2['y'] = griddata(data['x'], data['y'], xgrid)
    data2['x'] = xgrid
    # print(data2)
    # store the old data with the tag 'original' and remove th 'interp' tag from the new data 
    # data = data.rename(columns = {'a':'a_original','x':'x_original'})
    # data = data.rename(columns = {'a_interp': 'a', 'x_interp': 'x'})

    # do you want the old data? I don't recommend keeping it for this purpose, but I gave you the option.

    return data2
