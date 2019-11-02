import os
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report


if os.environ.get("DOCKER") == 'True':
    data = Path('/data/data')
else:
    data = Path('data')

output_dir = data / "output"
output_dir.mkdir(parents=True, exist_ok=True)

test = pd.read_csv(data / "test.csv").groupby('abbrev')

cl_acr = re.compile('^([a-zA-Z0-9]+)_([a-zA-Z0-9]+)\.csv$')

# get list of acronyms in output directory
acronyms = set([cl_acr.match(f.parts[-1]).group(2) for f in output_dir.glob("*.csv")])

# get a concatenated dataframe for each acronym
dataframes = {a:pd.concat((pd.read_csv(f, usecols=['predictions']) for f in output_dir.glob("*_{}.csv".format(a))), ignore_index=True, axis=1) for a in acronyms}

# get mode for each row and write out csv and classification details
for df in dataframes:
    dataframes[df] = dataframes[df].apply(pd.Series.mode, axis=1)
    dataframes[df].to_csv(output_dir / "voted_{}.csv".format(df), columns=[0], header=['predictions'])
    print("##" * 20)
    print(" " * 20 + df)
    print("##" * 20 + "\n")
    print(classification_report(test.get_group(df).expansion, dataframes[df][0]))
