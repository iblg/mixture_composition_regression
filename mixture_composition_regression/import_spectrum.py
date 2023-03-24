import pandas as pd

def clean_data(filename):
    """
    Open and clean a csv file of data.

    Parameters
    ----------
    filename : string
        Filepath to file.
        The refractive index of medium of origin and destination medium.

    Returns
    -------
    df : DataFrame
        Pandas DataFrame containing the x-ray absorption and

    Examples
    --------

    """

    df = pd.read_csv(filename)
    df = df.dropna(axis=1, how="all")

    row_cutoff = (
        df.isna().idxmax("index").where(df.isna().any(axis = "index"))
    )  # find a cutoff row by finding where there are any nan values
    row_cutoff = int(row_cutoff.mode()) - 1

    df = df.iloc[1:row_cutoff]  # cut off the row at that point
    df = df.astype("float")
    return df
