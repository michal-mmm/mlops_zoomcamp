import os
from datetime import datetime

import pandas as pd


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
df_input = pd.DataFrame(data, columns=columns)

LOCALSTACK_OPTIONS = {
'client_kwargs': {
    'endpoint_url': 'http://localhost:4566'
    }
}

year=2021
month=1

df_input.to_parquet(
    's3://nyc-duration/in/input_file.parquet',
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=LOCALSTACK_OPTIONS
)

os.system("python batch.py 2021 1")
