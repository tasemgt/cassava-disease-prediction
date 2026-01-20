Computer Vision Solution for Cassava Crop Disease Detection Problem
========
This project employs machine learning (Deep Learning) techniques for classifying & detecting illnesses in cassava crops for the purpose of better and improved agricultural profit.


Problem Description 
================
Cassava plantations are currently under threat from diseases. At present, detecting and diagnosing these diseases requires the presence of agricultural experts. Given that large-scale, commercialised plantations can be very huge, it would either require 
- A few experts available round the clock, or
- Multiple experts available for a short while.
Either way, expenses will become a major pain point.
Substituting human experts with a DL/ML model would make it such that farms can more easily monitor their cassava crops in real-time. This would provide a means of monitoring their cassava crops and make it easier to take swift actions in the event of disease detection.


About the Dataset
================
The data is contained in images (Cassava plant leaves) and are grouped in the following 5 classes:
- Cassava Bacterial Blight (CBB)
- Cassava Brown Steak Disease (CBSD)
- Cassava Green Mite (CGM)
- Cassava Mosaic Disease (CMD) and
- Healthy Cassava plant.


The Methodology
================
This project followed the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology with the following six important phases:
- Business Understanding
- Data Understanding
- Data Preparation
- Data Modelling
- Model Evaluation and
- Model Deployment


The Solution
================








Steps to Reproduce Project
================
1. **Prepare Code & Environment**
   
   - Clone the project repo and navigate to the directory
   - Open in vscode for better coding management & experience
     
     ```
     git clone https://github.com/tasemgt/passenger-booking-prediction.git
     cd passenger-booking-prediction
     ```
2. **Install dependencies**
   - Run the command `uv sync` to install dependencies
   - Navigate to the src directory `cd src`

3. **Start FastAPI Server Locally**
   - Inside the src folder, run the command `uvicorn predict:app --reload` to start the web service
   - Navigate to `127.0.0.1:8000/docs` to access FlaskAPI's service to run inference testing
     
4. **Start FastAPI via Docker**
   - From the root folder, run `docker build -t customer-booking-api .` to build the docker image
   - To start up the docker container, run `docker-compose up -d`
   - Navigate to `127.0.0.1:9696/docs` to access FlaskAPI's service to run inference testing
  
5. **Making Predictions (Inference)**
   - Use the following sample JSON customer data to test inference. You can tweak as you wish to see if customer completes booking or not.
   ```
     {
        "num_passengers": 2,
        "sales_channel": "Internet",
        "trip_type": "RoundTrip",
        "purchase_lead": 120,
        "length_of_stay": 10,
        "flight_hour": 14,
        "flight_day": "Tue",
        "route": "AKLDEL",
        "booking_origin": "New Zealand",
        "wants_extra_baggage": 1,
        "wants_preferred_seat": 0,
        "wants_in_flight_meals": 1,
        "flight_duration": 5.52
    }
   ```
   - You can try other routes like `AKLKUL`, `AKLKIX`, and countries like `Malaysia`, `Germany`, etc.
  

Contact
=======
Connect with me on LinkedIn ❤️ : [Michael Tase](https://www.linkedin.com/in/michael-tase-4151216a)

