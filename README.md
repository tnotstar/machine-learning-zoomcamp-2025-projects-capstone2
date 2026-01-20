# Machine Learning Zoomcamp (Cohort 2025)

## Capstone Project 2: A Stroke Predictor Service

### Project Description ###

Stroke is a medical condition in which poor blood flow to the brain causes cell death. According to the World Health Organization (WHO), stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. Early identification of high-risk individuals is critical for preventative healthcare. However, many risk factors—such as hypertension, heart disease, and lifestyle choices—interact in complex ways that are difficult for clinicians to quantify manually in a rapid screening environment.

#### Problem Statement

Medical professionals often lack automated, data-driven tools to assist in the triage of patients who are at high risk of stroke. While clinical data is frequently collected (e.g., BMI, glucose levels, smoking status), it is often stored without being leveraged for predictive analytics. There is a need for a reliable, scalable system that can process patient demographics and health indicators to provide a probability of stroke occurrence, allowing doctors to prioritize diagnostic tests and lifestyle interventions.

#### Objective

The objective of this project is to develop an end-to-end Machine Learning pipeline to predict whether a patient is likely to get a stroke based on input parameters such as gender, age, various diseases, and smoking status.

##### Key Goals:

* **Analyze and Prepare:** Perform extensive EDA on the "Stroke Prediction Dataset" to identify key drivers of stroke risk.

* **Model Development:** Train and tune multiple classification models (including linear models like Logistic Regression and tree-based models like Random Forest/XGBoost) to find the most accurate predictor.

* **Productionalize:** Package the best-performing model into a production-ready web service using Flask and Docker, enabling real-time predictions via a simple API.

#### Analysis and model selection

The code for analysis and model development is in the `submission.ipynb` file at the root of this repository.

> NOTE: The `submission.ipynb` file is a Jupyter notebook that contains the code for analysis and model development. It is not a Python script and cannot be run directly. To run the code, you will need to use a Jupyter notebook environment.

#### Running the assets in a local environment

This project use `uv` as the Python manager. To run the assets in a local environment, you will need to have `uv` installed. You can install `uv` by running the following command:

```bash
pip install uv
```

Once you have `uv` installed, you can run the assets in a local environment by running the following command:

```bash
uv sync
```

To run the Jupyter notebook, run the following command:

```bash
uv run jupyter-notebook
```

To train the model, run the following command:

```bash
uv run train.py
```

To run the FastAPI server, run the following command:

```bash
uv run predict.py
```

### Deploying the predictive model service with Google Build (CD/CI)

The predictive model can be trained using the script `train.py` and it was implemented as a *FastAPI* microservice in `predict.py`.  Also a Docker image is created through `Dockerfile` to be deployed at **Google Cloud Run**, following this steps:

#### Step 1: Go to the Console

Go to [console.cloud.google.com](https://console.cloud.google.com/) and log in.

1. If this is your first time, create a new **Project** (give it a name like "my-fastapi-demo").

2. You might be asked to enable **Billing**. Don't worry, remember the free tier exists, but Google needs a card to verify you are not a bot.

#### Step 2: Go to Cloud Run

1.  In the top search bar, type **"Cloud Run"** and select the first option.

2.  Click the blue **"CREATE SERVICE"** button at the top.

#### Step 3: Connect your GitHub

1.  Look for the section **"Deploy one revision from an existing container image"**. Instead of that, select the option below it:
    **Continuously deploy new revisions from a source repository**.

2.  Click the **"SET UP CLOUD BUILD"** button.

3.  Select **GitHub** as the provider.

4.  If you haven't done this before, authorize the *Google Cloud Build App* to access your GitHub account.

5.  **Repository:** Search for and select your repository (`your-user/your-project`).

6.  Click **Next**.

7.  **Build Configuration:**
    * Build Type: **Dockerfile**.
    * Source location: Keep it as `/your-path/Dockerfile` (assuming it's in the root folder).

8.  Click **Save**.

#### Step 4: Configure Public Access

You will return to the main creation screen. Scroll down to the **Authentication** section.

> **IMPORTANT:** Select **Allow unauthenticated invocations**.
>
> If you don't check this, your API will be private, and no one on the Internet will be able to access it without a Google key. Since you want to publish it, this must be enabled.

#### Step 5: Final Settings & Deploy

1.  Expand the **"Container, Networking, Security"** section (it's a dropdown arrow).

2.  Click on the **Settings** tab and verify that the **Container port** is set to `8080`.
    * Since your Dockerfile uses `${PORT:-8080}`, this will match perfectly.

3.  Click the blue **CREATE** button at the bottom.

#### What happens next?

1.  You will see a screen with metrics and logs. Google Cloud is downloading your code, reading your Dockerfile, and building the image.

2.  This will take **2-3 minutes** the first time.

3.  When it finishes, you will see a green checkmark, and a **URL** ending in `.run.app` will appear at the top.

### Deployed Predictor Service

Following URL is the documentation of deployed service:

> https://machine-learning-zoomcamp-428800185377.europe-west1.run.app/docs

The following code is a simple Python segment to test the service from anywhere:

```python
import requests

input = {
    "gender": "Male",
    "age": 67.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": 1,
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}

res = requests.post("https://machine-learning-zoomcamp-428800185377.europe-west1.run.app/predict", json=input)
res.raise_for_status()

print(res.json())
```