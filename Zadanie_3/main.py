import torch
import numpy as np
import pandas as pd
import sklearn.datasets as datasets

if __name__ == "__main__":
    ds = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[ds['data'], ds['target']], columns=[ds["feature_names"] + ['target']])
    print(data.to_markdown(index=False))
