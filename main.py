from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from pipeline import CatAttrEncoder
import util  
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    util.load_saved_artifacts()

class DSJobsInput(BaseModel):
    work_year: int
    experience_level: str
    employment_type: str
    job_title: str
    salary_currency: str
    employee_residence: str
    remote_ratio: int
    company_location: str
    company_size: str

@app.get("/")
async def hello(name: str = "Karisa"):
    return {"message": f"Welcome {name}"}

@app.post("/predict-dsjobs-salaries")
async def predict_dsjobs_salary(dsjobs: DSJobsInput):
    job_details = dsjobs.dict()
    
    estimated_salary = util.get_estimated_salary(job_details)
    
    return JSONResponse(content={"estimated_salary": estimated_salary})


@app.get("/get_job_titles")
def get_job_titles():
    return JSONResponse(content={"job_titles": util.get_job_titles()})

@app.get("/get_company_locations")
def get_company_locations():
    return JSONResponse(content={"company_locations": util.get_company_locations()})

@app.get("/get_currency")
def get_currency():
    return JSONResponse(content={"currency": util.get_currency()})

@app.get("/get_employee_residence")
def get_employee_residence():
    return JSONResponse(content={"employee_residence": util.get_employee_residence()})

@app.get("/get_work_year")
def get_work_year():
    return JSONResponse(content={"work_year": util.get_work_year()})