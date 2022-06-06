import os
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
import ramda as R
import requests
import yaml
from joblib import Memory, memory
from psycopg2 import connect
from psycopg2.sql import SQL, Identifier, Literal, Composed

notebook_path = os.path.abspath("Detection_model.ipynb")

cachedir = "./db_cache"
Path(cachedir).mkdir(exist_ok=True)
mem = Memory(location=cachedir, verbose=1)
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
    credentials = load_yaml_file(
        Path(notebook_path).with_name("postgres_credentials.yaml")
    )
    connection = connect(**credentials)
    res = pd.read_sql(query, connection)
    connection.close()
    return res


vital_ids = {"rhr": 65, "steps": 9, "sleep_duration": 43}


def load_standardized_vitals(vital_type: str):
    return pull_from_postgres(
        SQL(
            """
        SELECT
            signal {column_name}, test_week_start, user_id
        FROM
            jakob.ml_vital_features
        WHERE
            type={vital_type} AND
            baseline_count>30 AND
            signal_count > 3
        """
        ).format(
            column_name=Identifier(f"{vital_type}_metric"),
            vital_type=Literal(vital_ids[vital_type]),
        )
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


url = "https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Daten/VOC_VOI_Tabelle.xlsx?__blob=publicationFile"
omicronba2_column = "Omikron_BA.2+BA.2.1+BA.2.2+BA.2.3+BA.2.3.1+BA.2.3.2+BA.2.3.3+BA.2.3.4+BA.2.4+BA.2.5+BA.2.6+BA.2.7+BA.2.8+BA.2.9+BA.2.9.1+BA.2.9.2+BA.2.10+BA.2.10.1+BA.2.11+BA.2.12+BA.2.12.1+BA.2.13+BA.2.14+BA.2.15+BA.2.16+BA.2.17+BA.2.18+BA.2.19+BA.2.20+BA.2.21+BA.2.22+BA.2.23+BA.2.24+BA.2.25+BA.2.25.1+BA.2.26+BA.2.27+BA.2.28+BA.2.29+BA.2.30+BA.2.31+BA.2.32+BA.2.33+BA.2.34_Anteil (%)"
omicronba1_column = "Omikron_BA.1+BA.1.1+BA.1.1.1+BA.1.1.2+BA.1.1.3+BA.1.1.4+BA.1.1.5+BA.1.1.6+BA.1.1.7+BA.1.1.8+BA.1.1.9+BA.1.1.10+BA.1.1.11+BA.1.1.12+BA.1.1.13+BA.1.1.14+BA.1.1.15+BA.1.1.16+BA.1.1.17+BA.1.1.18+BA.1.2+BA.1.3+BA.1.4+BA.1.5+BA.1.6+BA.1.7+BA.1.8+BA.1.9+BA.1.10+BA.1.12+BA.1.13+BA.1.13.1+BA.1.14+BA.1.14.1+BA.1.14.2+BA.1.15+BA.1.15.1+BA.1.15.2+BA.1.16+BA.1.16.1+BA.1.16.2+BA.1.17+BA.1.17.1+BA.1.17.2+BA.1.18+BA.1.19+BA.1.20+BA.1.21+BA.1.21.1+BA.1.22_Anteil (%)"
delta_column = "Delta_AY.1+AY.2+AY.3+AY.3.1+AY.3.2+AY.3.3+AY.3.4+AY.4+AY.4.1+AY.4.2+AY.4.2.1+AY.4.2.2+AY.4.2.3+AY.4.2.4+AY.4.2.5+AY.4.3+AY.4.4+AY.4.5+AY.4.6+AY.4.7+AY.4.8+AY.4.9+AY.4.10+AY.4.11+AY.4.12+AY.4.13+AY.4.14+AY.4.15+AY.4.16+AY.4.17+AY.5+AY.5.1+AY.5.2+AY.5.3+AY.5.4+AY.5.5+AY.5.6+AY.5.7+AY.6+AY.7+AY.7.1+AY.7.2+AY.8+AY.9+AY.9.2+AY.9.2.1+AY.9.2.2+AY.10+AY.11+AY.13+AY.14+AY.15+AY.16+AY.16.1+AY.17+AY.18+AY.19+AY.20+AY.20.1+AY.21+AY.22+AY.23+AY.23.1+AY.23.2+AY.24+AY.24.1+AY.25+AY.25.1+AY.25.1.1+AY.25.1.2+AY.25.2+AY.25.3+AY.26+AY.26.1+AY.27+AY.28+AY.29+AY.29.1+AY.29.2+AY.30+AY.31+AY.32+AY.33+AY.33.1+AY.33.2+AY.34+AY.34.1+AY.34.1.1+AY.34.2+AY.35+AY.36+AY.36.1+AY.37+AY.38+AY.39+AY.39.1+AY.39.1.1+AY.39.1.2+AY.39.1.3+AY.39.1.4+AY.39.2+AY.39.3+AY.40+AY.41+AY.42+AY.42.1+AY.43+AY.43.1+AY.43.2+AY.43.3+AY.43.4+AY.43.5+AY.43.6+AY.43.7+AY.43.8+AY.43.9+AY.44+AY.45+AY.46+AY.46.1+AY.46.2+AY.46.3+AY.46.4+AY.46.5+AY.46.6+AY.46.6.1+AY.47+AY.48+AY.49+AY.50+AY.51+AY.52+AY.53+AY.54+AY.55+AY.56+AY.57+AY.58+AY.59+AY.60+AY.61+AY.62+AY.63+AY.64+AY.65+AY.66+AY.67+AY.68+AY.69+AY.70+AY.71+AY.72+AY.73+AY.74+AY.75+AY.75.2+AY.75.3+AY.76+AY.77+AY.78+AY.79+AY.80+AY.81+AY.82+AY.83+AY.84+AY.85+AY.86+AY.87+AY.88+AY.90+AY.91+AY.91.1+AY.92+AY.93+AY.94+AY.95+AY.98+AY.98.1+AY.99+AY.99.1+AY.99.2+AY.100+AY.101+AY.102+AY.102.1+AY.102.2+AY.103+AY.103.1+AY.103.2+AY.104+AY.105+AY.106+AY.107+AY.108+AY.109+AY.110+AY.111+AY.112+AY.112.1+AY.112.2+AY.112.3+AY.113+AY.114+AY.116+AY.116.1+AY.117+AY.118+AY.119+AY.119.1+AY.119.2+AY.120+AY.120.1+AY.120.2+AY.120.2.1+AY.121+AY.121.1+AY.122+AY.122.1+AY.122.2+AY.122.3+AY.122.4+AY.122.5+AY.122.6+AY.123+AY.123.1+AY.124+AY.124.1+AY.124.1.1+AY.125+AY.125.1+AY.126+AY.127+AY.127.1+AY.127.2+AY.127.3+AY.128+AY.129+AY.131+AY.132+AY.133+AY.134+B.1.617.2_Anteil (%)"


def add_variant_data(feature_data: pd.DataFrame) -> pd.DataFrame:
    r = requests.get(url)
    with open("tmp.xlsx", "wb") as file:
        file.write(r.content)

    variant_data = pd.read_excel("tmp.xlsx", sheet_name="VOC", skipfooter=1)

    # parse calendar week to date
    variant_data["test_week_start"] = variant_data["KW"].apply(
        R.pipe(
            R.replace("K", ""),
            lambda d: datetime.strptime(d + "-1", "%Y-W%W-%w").date(),
        )
    )

    variant_data["test_week_start"] = variant_data["test_week_start"].apply(
        pd.to_datetime
    )

    # add one entry for today assuming the variant shares don't change
    variant_data = pd.concat(
        [
            variant_data,
            pd.DataFrame(
                {
                    "test_week_start": [pd.to_datetime(datetime.now().date())],
                    delta_column: [0],
                    omicronba1_column: [1],
                    omicronba2_column: [99],
                }
            ),
        ],
        ignore_index=True,
    )

    # upsample variant data to daily values, using linear interpolation
    upsampled_variant_data = (
        variant_data[
            ["test_week_start", omicronba1_column, omicronba2_column, delta_column]
        ]
        .rename(
            columns={
                omicronba1_column: "omicronba1_share",
                omicronba2_column: "omicronba2_share",
                delta_column: "delta_share",
            }
        )
        .set_index("test_week_start")
        .resample("D")
        .interpolate(method="linear")
        .reset_index()
    )

    # merge omicron share on other data
    return upsampled_variant_data.merge(feature_data, on="test_week_start", how="outer")


def loading_and_pre_processing_pipeline():
    """
    calcualte metrics from rhr and steps and join with test results and reported symptoms.

    Date in test results and symptoms data is the first day of the week for which test and symptoms were reported.
    Dates for baseline are 60 days prior to the test week,
    Dates for signal in rhr and steps are the 7 days during the week for which the test was reported.
    """
    rhr_metric = load_standardized_vitals(
        "rhr",
    ).set_index(["user_id", "test_week_start"])

    steps_metric = load_standardized_vitals(
        "steps",
    ).set_index(["user_id", "test_week_start"])

    test_results = load_test_results_symptoms_sex_age().set_index(
        ["user_id", "test_week_start"]
    )

    feature_data = rhr_metric.join(steps_metric).join(test_results).reset_index()

    feature_data["test_week_start"] = feature_data["test_week_start"].apply(
        pd.to_datetime
    )

    del rhr_metric
    del steps_metric
    del test_results

    return add_variant_data(feature_data)
