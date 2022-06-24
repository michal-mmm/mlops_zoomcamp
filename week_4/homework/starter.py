import argparse
import pickle

import numpy as np
import pandas as pd


def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse year and month for FHV trip data."
    )
    parser.add_argument("--year", help="year", required=True, type=int, dest="YEAR")
    parser.add_argument("--month", help="month", required=True, type=int, dest="MONTH")
    args = parser.parse_args()
    return args.YEAR, args.MONTH


def main():
    categorical = ["PUlocationID", "DOlocationID"]
    # parse arguments
    YEAR, MONTH = parse_arguments()
    OUTPUT_PATH = f"results/{YEAR:04d}_{MONTH:02d}_predictions.parquet"

    # read/transform data
    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(
        f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{YEAR:04d}-{MONTH:02d}.parquet",
        categorical,
    )
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)

    # predictions
    y_pred = lr.predict(X_val)
    print(np.mean(y_pred))

    # save to a folder
    df["ride_id"] = f"{YEAR:04d}/{MONTH:02d}_" + df.index.astype("str")
    df["pred"] = y_pred
    df_result = df[["ride_id", "pred"]]
    df_result.to_parquet(OUTPUT_PATH, engine="pyarrow", compression=None, index=False)


if __name__ == "__main__":
    main()
