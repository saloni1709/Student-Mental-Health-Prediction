# ============================================================
# IMPORTS
# ============================================================

# Joblib ka use saved/trained ML model ko load karne ke liye
import joblib

# Pandas ka use input data ko DataFrame me convert karne ke liye
import pandas as pd

# FastAPI application banane ke liye
from fastapi import FastAPI

# Pydantic input validation ke liye
from pydantic import BaseModel, Field

# Fixed/limited values define karne ke liye
from typing import Literal

# Frontend aur backend ke beech request allow karne ke liye
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

# Saved .pkl file se trained ML pipeline/model load kar rahe hain
#
# IMPORTANT:
# Ye filename tumhare project folder me available .pkl file
# ke exact naam ke according hai.
model = joblib.load(
    'Student_Social_Media_And_Mental_Health_Impact.pkl'
)


# ============================================================
# TOP COUNTRIES
# ============================================================

# Notebook me top 10 countries ko select kiya gaya tha.
#
# Agar country in countries me hai,
# to uska original country name use hoga.
#
# Agar country in countries me nahi hai,
# to usse "Other" category me convert karenge.

top_countries = [
    'India',
    'USA',
    'Canada',
    'Australia',
    'UK',
    'Germany',
    'Mexico',
    'Turkey',
    'France'
]


# ============================================================
# CREATE FASTAPI APP
# ============================================================

# FastAPI application create kar rahe hain
app = FastAPI()


# ============================================================
# CORS MIDDLEWARE
# ============================================================

# CORS frontend ko backend API access karne ki permission deta hai.
#
# allow_origins=["*"]
# ka matlab kisi bhi origin se request allow hai.
app.add_middleware(
    CORSMiddleware,

    # Sabhi origins ko allow kar rahe hain
    allow_origins=["*"],

    # Sabhi HTTP methods allow kar rahe hain
    # GET, POST, PUT, DELETE etc.
    allow_methods=["*"],

    # Sabhi headers allow kar rahe hain
    allow_headers=["*"]
)


# ============================================================
# INPUT DATA MODEL
# ============================================================

# Student se API me jo input data receive hoga,
# uska structure yahan define kar rahe hain.
class StudentInput(BaseModel):

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    # Student ki age integer honi chahiye.
    #
    # ge = greater than or equal to
    # le = less than or equal to
    #
    # Isliye age 10 se 100 ke beech honi chahiye.
    age: int = Field(
        ...,
        ge=10,
        le=100
    )


    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    # Gender sirf Male ya Female ho sakta hai.
    gender: Literal[
        "Male",
        "Female"
    ]


    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    # Country ka naam string/text format me hoga.
    #
    # Note:
    # Country input user se lenge,
    # lekin model ko directly Country nahi bhejenge.
    #
    # Model ko sirf Grouped_country bhejna hai.
    country: str


    # --------------------------------------------------------
    # ACADEMIC LEVEL
    # --------------------------------------------------------

    # Academic level sirf given options me se ek hona chahiye.
    academic_level: Literal[
        "Undergraduate",
        "Graduate",
        "High School"
    ]


    # --------------------------------------------------------
    # MOST USED PLATFORM
    # --------------------------------------------------------

    # Student ka most-used social media platform.
    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat"
    ]


    # --------------------------------------------------------
    # PURPOSE OF USE
    # --------------------------------------------------------

    # Student social media ka main purpose kya hai.
    purpose_of_use: Literal[
        "Networking",
        "Education",
        "Entertainment",
        "News"
    ]


    # --------------------------------------------------------
    # AVERAGE DAILY USAGE HOURS
    # --------------------------------------------------------

    # Daily social media usage hours.
    #
    # Float isliye use kiya hai kyunki value decimal ho sakti hai.
    # Example: 3.5 hours
    #
    # Value 0 se 24 ke beech honi chahiye.
    avg_daily_usage_hours: float = Field(
        ...,
        ge=0,
        le=24
    )


    # --------------------------------------------------------
    # DAILY UNLOCKS
    # --------------------------------------------------------

    # Ek din me phone/social media kitni baar unlock kiya.
    #
    # Ye count hai, isliye integer use kiya hai.
    #
    # Negative value allowed nahi hai.
    daily_unlocks: int = Field(
        ...,
        ge=0
    )


    # --------------------------------------------------------
    # STUDY HOURS
    # --------------------------------------------------------

    # Student daily kitne hours study karta hai.
    #
    # Decimal value allowed hai.
    # Example: 4.5
    study_hours: float = Field(
        ...,
        ge=0,
        le=24
    )


    # --------------------------------------------------------
    # PHYSICAL ACTIVITY HOURS
    # --------------------------------------------------------

    # Student daily kitne hours physical activity karta hai.
    physical_activity_hours: float = Field(
        ...,
        ge=0,
        le=24
    )


    # --------------------------------------------------------
    # SLEEP HOURS
    # --------------------------------------------------------

    # Student per night kitne hours sota hai.
    sleep_hours_per_night: float = Field(
        ...,
        ge=0,
        le=24
    )


    # --------------------------------------------------------
    # STRESS LEVEL
    # --------------------------------------------------------

    # Stress level sirf given categories me se ek hona chahiye.
    stress_level: Literal[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]


# ============================================================
# RESPONSE DATA MODEL
# ============================================================

# API client ko jo response bhejna hai,
# uska structure yahan define kar rahe hain.
class PredictionResponse(BaseModel):

    # Predicted Mental Health Score.
    #
    # Example:
    # 6.78
    predicted_mental_health_score: float


# ============================================================
# HOME / ROOT API
# ============================================================

# "/" URL par GET request handle karega.
@app.get('/')
def greet():

    # Simple welcome message return karega.
    return {
        "message": "Welcome to Student Mental Health Prediction API"
    }


# ============================================================
# PREDICTION API
# ============================================================

# "/predict" URL par POST request handle karega.
#
# response_model:
# API response PredictionResponse ke format me hona chahiye.
@app.post(
    '/predict',
    response_model=PredictionResponse
)
def predict(data: StudentInput):


    # ========================================================
    # COUNTRY GROUPING
    # ========================================================

    # Agar user ka country top_countries me present hai,
    # to wahi country name use hoga.
    #
    # Agar country top_countries me nahi hai,
    # to usse "Other" category me convert karenge.

    country_group = (
        data.country
        if data.country in top_countries
        else "Other"
    )


    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    # User ke input ko Pandas DataFrame me convert kar rahe hain.
    #
    # IMPORTANT:
    # Yahan wahi features hone chahiye
    # jo training ke time model ko diye gaye the.
    #
    # Training notebook me features the:
    #
    # Age
    # Gender
    # Academic_Level
    # Most_Used_Platform
    # Purpose_Of_Use
    # Avg_Daily_Usage_Hours
    # Daily_Unlocks
    # Study_Hours
    # Physical_Activity_Hours
    # Sleep_Hours_Per_Night
    # Stress_Level
    # Grouped_country
    #
    # "Country" directly model ko nahi dena hai.

    input_row = pd.DataFrame([{

        # Student ki age
        'Age': data.age,

        # Student ka gender
        'Gender': data.gender,

        # Academic level
        'Academic_Level': data.academic_level,

        # Most-used social media platform
        'Most_Used_Platform': data.most_used_platform,

        # Social media use karne ka main purpose
        'Purpose_Of_Use': data.purpose_of_use,

        # Average daily social media usage
        'Avg_Daily_Usage_Hours': data.avg_daily_usage_hours,

        # Daily phone/social media unlock count
        'Daily_Unlocks': data.daily_unlocks,

        # Daily study hours
        'Study_Hours': data.study_hours,

        # Daily physical activity hours
        'Physical_Activity_Hours': data.physical_activity_hours,

        # Per night sleep hours
        'Sleep_Hours_Per_Night': data.sleep_hours_per_night,

        # Stress level
        'Stress_Level': data.stress_level,

        # Country ko top country ya "Other" group me convert kiya
        'Grouped_country': country_group
    }])


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    # Trained ML pipeline se prediction kar rahe hain.
    #
    # model.predict() normally array return karta hai.
    #
    # Example:
    #
    # [6.777777]
    #
    # [0] se first prediction value nikal rahe hain.

    prediction = model.predict(input_row)[0]


    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    # Prediction ko float me convert kar rahe hain.
    #
    # round(..., 2)
    # prediction ko 2 decimal places tak round karega.
    #
    # Example:
    #
    # 6.777777 → 6.78

    return PredictionResponse(
        predicted_mental_health_score=round(
            float(prediction),
            2
        )
    )