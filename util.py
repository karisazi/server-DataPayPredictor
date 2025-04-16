import pickle
import joblib
import json
from pipeline import CatAttrEncoder
import pandas as pd
from datetime import datetime

pd.set_option('future.no_silent_downcasting', True)

__job_titles = None
__company_locations = None
__currency = None
__employee_residence = None
__data_columns = None
__model = None
__feature_transformation_pipeline = None


def get_estimated_salary(job_details: dict):
    job_details_df = pd.DataFrame([job_details])
    transformed_job_details_df = __feature_transformation_pipeline.transform(job_details_df)
    prediction = __model.predict(transformed_job_details_df)
    return float(round(prediction[0], 2))


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __job_titles
    global __company_locations
    global __currency
    global __employee_residence

    with open("./artifacts/columns.json", "r") as f:
        data = json.load(f)
        __data_columns = data['data_columns']
        __job_titles = data['job_titles']
        __company_locations = data['company_locations']
        __currency = data['currency']
        __employee_residence = data['employee_residence']

        
    global __model
    if __model is None:
        with open('./artifacts/ds_jobs_salaries_model.pickle', 'rb') as f:
            __model = pickle.load(f)
            
    global __feature_transformation_pipeline
    if __feature_transformation_pipeline is None:
        with open('./artifacts/feature_transformation_pipeline.pkl', 'rb') as f:
            __feature_transformation_pipeline = joblib.load(f)
            
    print("loading saved artifacts...done")


def get_data_columns():
    return __data_columns

def get_job_titles():
    return __job_titles

def get_company_locations():
    return __company_locations

def get_currency():
    return __currency

def get_employee_residence():
    return __employee_residence

def get_work_year():
    current_year = datetime.now().year
    work_year = [year for year in range(2020, current_year+1)]
    return work_year
        
    

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_estimated_salary({
        "work_year": 2025,
        "experience_level": "EX",
        "employment_type": "FT",
        "job_title": "Data Scientist",
        "salary_currency": "USD",
        "employee_residence": "US",
        "remote_ratio": 100,
        "company_location": "US",
        "company_size": "M"
        }
        ))
    print(__job_titles)
    print(get_work_year())
