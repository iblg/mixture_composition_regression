import pandas as pd


def clean_data(filename: str, dropna: str = 'all') -> pd.DataFrame:
    """
    Open and clean a csv file of data.

    Parameters
    ----------
    filename : str
        Filepath to file.
        The refractive index of medium of origin and destination medium.

    Returns
    -------
    df : DataFrame
        Pandas DataFrame containing the x-ray absorption and

    Examples
    --------
    :param dropna: str
        Default 'all': drops all values in a row if entire row is composed of nan values.
        Specify dropna='any' to drop all values in any row is composed of nan values.

    """

    df = pd.read_csv(filename)

    df = df.dropna(axis=1, how=dropna)

    # row_cutoff = (
    #     df.isna().idxmax("index").where(df.isna().any(axis="index"))
    # )  # find a cutoff row by finding where there are any nan values
    # mask = df.isna().any(axis=1)
    # print(mask)
    # # #    print(type(row_cutoff))
    # print(row_cutoff.index)
    # print('Row_cutoff: {}'.format(row_cutoff))
    # row_cutoff = int(row_cutoff.mode()) - 1
    #
    # df = df.iloc[1:row_cutoff]  # cut off the row at that point
    df = df.astype("float")
    return df
