from datetime import datetime

import batch
import pandas as pd


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)
    print(df)

    categorical = ['PUlocationID', 'DOlocationID']
    actual_df = batch.prepare_data(df, 2021, 1, categorical)
    
    output_data = [
        ('-1', '-1', dt(1, 2), dt(1, 10), 8.0, '2021/01_0'),
        ('1', '1', dt(1, 2), dt(1, 10), 8.0, '2021/01_1')
    ]
    expected_df = pd.DataFrame(output_data, columns=columns+['duration', 'ride_id'])

    pd.testing.assert_frame_equal(actual_df, expected_df)