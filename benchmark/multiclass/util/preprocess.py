import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_and_get_dataset(path):
    # load and preprocess optdigits data
    optdigits_tra = pd.read_csv(path + "optdigits.tra", header=None)
    optdigits_tes = pd.read_csv(path + "optdigits.tes", header=None)
    optdigits_df = pd.concat([optdigits_tra, optdigits_tes]).reset_index(drop=True)

    optdigits_X = optdigits_df.iloc[:, range(64)].to_numpy()
    optdigits_y = optdigits_df.iloc[:, 64].to_numpy()
    optdigits = (optdigits_X, optdigits_y)

    # load and preprocess pendigits data
    pendigits_tra = pd.read_csv(path + "pendigits.tra", header=None)
    pendigits_tes = pd.read_csv(path + "pendigits.tes", header=None)
    pendigits_df = pd.concat([pendigits_tra, pendigits_tes]).reset_index(drop=True)

    pendigits_X = pendigits_df.iloc[:, range(16)].to_numpy()
    pendigits_y = pendigits_df.iloc[:, 16].to_numpy()
    pendigits = (pendigits_X, pendigits_y)

    # load and preprocess satimage data
    satimage_tra = pd.read_csv(path + "sat.trn", sep=" ", header=None)
    satimage_tes = pd.read_csv(path + "sat.tst", sep=" ", header=None)
    satimage_df = pd.concat([satimage_tra, satimage_tes]).reset_index(drop=True)

    satimage_X = satimage_df.iloc[:, range(36)].to_numpy()
    satimage_y = satimage_df.iloc[:, 36].to_numpy()
    satimage_y = np.where(satimage_y == 7, 0, satimage_y)
    satimage = (satimage_X, satimage_y)

    return optdigits, pendigits, satimage
