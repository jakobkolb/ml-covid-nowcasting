
import os
from datetime import date, datetime
from pathlib import Path
from typing import Union

import pandas as pd
import yaml
from joblib import Memory, memory
from psycopg2 import connect
from psycopg2.sql import SQL, Identifier, Literal, Composed

notebook_path = os.path.abspath("Detection model (Logistic regression).ipynb")

cachedir = "./db_cache"
Path(cachedir).mkdir(exist_ok=True)
mem = Memory(location=cachedir, verbose=0)
memory._build_func_identifier = lambda func: func.__name__


def parse_yyyy_mm_dd(datestring: str) -> date:
    return datetime.strptime(datestring, "%Y-%m-%d").date()


def load_yaml_file(path) -> dict:
    with open(path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


@mem.cache
def pull_from_postgres(query: Union[str, SQL, Composed]) -> pd.DataFrame:
    credentials = load_yaml_file(Path(notebook_path).with_name("postgres_credentials.yaml"))
    connection = connect(**credentials)
    res = pd.read_sql(query, connection)
    connection.close()
    return res


vital_ids = {'rhr': 65, 'steps': 9, 'sleep_duration': 43}

def load_standardized_vitals(vital_type: str, baseline_type: str, signal_type: str):
    return pull_from_postgres(
        SQL(
            """
            SELECT
                *
            FROM
                (SELECT
                    (signal.{signal_type} - baseline.{baseline_type}) / stats.std {column_name},
                    features.user_id, features.test_week_start
                FROM
                    datenspende_derivatives.daily_vital_rolling_window_time_series_features baseline,
                    datenspende_derivatives.daily_vital_rolling_window_time_series_features signal,
                    datenspende_derivatives.daily_vital_statistics_before_infection stats,
                    (
                        SELECT
                            *
                        FROM datenspende_derivatives.homogenized_features
                        ORDER BY user_id, test_week_start
                    ) features
                WHERE
                    -- match vital types
                    baseline.type = {vital_type} AND
                    baseline.type = signal.type AND
                    baseline.type = stats.type AND
                    -- match source
                    baseline.source = signal.source AND
                    baseline.source = stats.source AND
                    -- match user_ids
                    features.user_id = baseline.user_id AND
                    features.user_id = signal.user_id AND
                    features.user_id = stats.user_id AND
                    -- match dates
                    features.test_week_start - integer '4' = baseline.date AND
                    features.test_week_start + integer '4' = signal.date AND
                    -- dont devide by zero
                    stats.std > 0
                ORDER BY user_id, test_week_start) vitals
            WHERE
                -- only download not null values
                {column_name} IS NOT NULL
            """
        ).format(column_name=Identifier(f'{vital_type}_metric'), vital_type=Literal(vital_ids[vital_type]),
                 baseline_type=Identifier(baseline_type), signal_type=Identifier(signal_type))
    )


def load_test_results_symptoms_sex_age(*_) -> pd.DataFrame:
    test_results = pull_from_postgres(
        SQL(
            """
            SELECT
                test_week_start,
                user_id,
                administered_vaccine_doses as vaccination_status,
                days_since_last_dose,
                f40 as chills,
                f41 as body_pain,
                f42 as loss_of_taste_and_smell,
                f43 as fatigue,
                f44 as cough,
                f45 as cold,
                f46 as diarrhea,
                f47 as sore_throat,
                f49 as asymptomatic,
                f10 as test_result,
                f76 as fittness,
                f127 as sex,
                f133 as age

            FROM
                datenspende_derivatives.homogenized_features
            -- WHERE test_week_start > '2022-01-01'
            """
        )
    )
    test_results["date"] = pd.to_datetime(
        test_results["test_week_start"], format="%Y-%m-%d"
    )
    return test_results

def loading_and_pre_processing_pipeline():
    """
    calcualte metrics from rhr and steps and join with test results and reported symptoms.

    Date in test results and symptoms data is the first day of the week for which test and symptoms were reported.
    Dates for baseline are 60 days prior to the test week,
    Dates for signal in rhr and steps are the 7 days during the week for which the test was reported.
    """

    rhr_metric = load_standardized_vitals('rhr', 'fiftysix_day_median_min_30_values',
                                          'seven_day_max_min_3_values').set_index(['user_id', 'test_week_start'])

    steps_metric = load_standardized_vitals('steps', 'fiftysix_day_median_min_30_values',
                                            'seven_day_mean_min_3_values').set_index(['user_id', 'test_week_start'])

    sleep_metric = load_standardized_vitals('sleep_duration', 'fiftysix_day_median_min_30_values',
                                            'seven_day_mean_min_3_values').set_index(['user_id', 'test_week_start'])

    test_results = load_test_results_symptoms_sex_age().set_index(['user_id', 'test_week_start'])

    return rhr_metric.join(steps_metric).join(sleep_metric).join(test_results).reset_index()