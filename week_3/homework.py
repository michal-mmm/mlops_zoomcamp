import datetime
from dateutil.relativedelta import relativedelta
import pickle

import pandas as pd
from prefect import flow, task, get_run_logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    # it's stupid, I have to create logger in each task ...
    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):
    logger = get_run_logger()
    logger.info(f'The date is: {date}')

    # prev month, and prev.prev month
    train_date = date + relativedelta(months=-2)
    val_date = date + relativedelta(months=-1)

    # f-strings
    train_path = f'./data/fhv_tripdata_{train_date.year}-{train_date.month:02d}.parquet'
    val_path = f'./data/fhv_tripdata_{val_date.year}-{val_date.month:02d}.parquet'
    logger.info(f'Training path: {train_path}')
    logger.info(f'Validation path: {val_path}')
    return train_path, val_path

# just a flow decorator is enough to bring it into prefect
# in prefect 1.0 everything inside main must have been inside task
# in prefect 2.0 you can mix tasks and non tasks
# if you are mixing NATIVE PYTHON and PREFECT TASKS, by calling the task
# you make a future, and by calling the future we HAVE TO CALL THAT RESULT
@flow
def main(date=None):
    if not date:
        date = datetime.date.today()
    else:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
    
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # save vectorizer and model for future use
    model_path = f'models/model-{date:%y-%m-%d}.bin'
    dv_path = f'models/dv-{date:%y-%m-%d}.b'
    # pickle is not really best format for that, but that's ok for now
    with open (model_path, 'wb') as f_out:
        pickle.dump(lr, f_out)
    with open(dv_path, 'wb') as f_out:
        pickle.dump(dv, f_out)

# main(date="2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name='model_training_lr',
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    # this is required for LOCAL STORAGE runs
    flow_runner=SubprocessFlowRunner(),
    # tags are metadata, you can do filtering in the dashboard
    # or specify "this compute like 'gpu1' to execute this on given machine"
    tags=['ml']
)