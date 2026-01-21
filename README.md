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
An End-to-End machine learning model that classifies cassava plant images as healthy or unhealthy into the different classes of diseases as listed above.
The solution has the following subsections:

1. **Tech Stack** <br>
   1.1. *Libraries*
   - PyTorch - A python wrapper of a C/C++ library for machine learning and deep learning. It provides a lot of utilities and implementations for a variety of machine learning ideas such as neural networks, NLP, normalization layers, and others).
   - torchvision - A subset of the PyTorch ecosystem, with specific focus on images and computer vision providing datasets, pretrained models and more.
   - Pillow/PIL (Python Image Libary) - A Python library desined for the purpose of image loading and manipulation.
   - Matplotlib - A Python library for data visualization.
   - imageio – Library for reading and writing a wide range of image and video formats.
   - onnx (Open Neural Network Exchange) – Open format for representing machine learning models across different frameworks.
   - onnxruntime – High-performance inference engine for executing ONNX machine learning models.
   - onnxscript – Tool for authoring, inspecting, and transforming ONNX models using Python syntax.
   - pandas – Data analysis and manipulation library providing DataFrame and Series structures.
   - numpy – Core library for numerical computing with support for arrays, matrices, and mathematical operations.
   - requests – Simple and user-friendly HTTP library for making web requests.
  
   1.2. *Tools & Environment*
   - Docker – Platform for packaging, distributing, and running applications in isolated containers.
   - AWS ECR (Elastic Container Registry) – Managed AWS service for storing, managing, and deploying Docker container images.
   - AWS Lambda – Serverless compute service that runs code in response to events without managing servers.
   - uv – Ultra-fast Python package installer and dependency manager written in Rust, compatible with pip and virtual environments.
  
2. **EDA (Exploratory Data Analysis)** <br>
   EDA is the process of sifting through data with the goal of extracting insights. These insights allow a better understanding of the available data and what can be done with it.
   They can also be used for guided preparation of the dataset in the appropriate manner. Just like regular analysis, EDA begins with a set of __questions__ and/or __hypotheses__.
   The EDA process will then prove or disprove these hypotheses, and then help reveal other points of inquiry along the way.

   After EDA, it was observed that there are:
   - Indeed five categories of cassava images.
   - The images are generally large in size. This would imply that a lot of computation will take place.
   - The sizes of the images vary. This would require us to ensure that the images are of the same size.
   - The image classes are imbalanced. This might require the use of specialized metrics for evaluation, such as ROC AUC. Also, by leveraging the use pretrained models, We can try to bypass this issue of class imbalance. 
   <br>
   <img width="600" height="474" alt="Screenshot 2026-01-21 at 11 25 14" src="https://github.com/user-attachments/assets/c4e68699-9241-4989-a2ad-69eee5a6240e" />
   <br>
   **Fig:** *A snapshot of some classes and images present in the dataset*

3. **Data Modelling** <br>
   The following is a highlighted overview of steps and items used in data preparation and modelling
   - Loading pretrained model
   - Reconfigure pretrained model.
      + VGG-13
      + Resnet-18
   - Model weights were initialised.
   - Instantiate training utilities like the _optimizer_.
   - Wire up the training loop.
   - Train the model using VGG-13 and Resnet-18 pre-trained models using finetuned and frozen weights for both respectively.
   - Model is trained on a CUDA machine for improved efficiency in process.
   - Hyperparameter tunning (Truning the learning rate, using Adam optimiser, and carrying out data augmentation to improve accuracy).
  
4. **Evaluation & Model Selection** <br>
   After training, the Resnet-18 (fine-tuned) model proved to be the best model in terms of performance having much more higher accuracy and lower loss as training epochs increase offering better generalisation on test data.
   <br>
   <img width="962" height="412" alt="Screenshot 2026-01-21 at 11 28 46" src="https://github.com/user-attachments/assets/d3d40c4d-df9c-42a1-b369-14aa0ca6867e" />
   <br>
   **Fig:** *A snapshot of evaluated Resnet-18 fine-tuned model*

5. **Model Deployment** <br>
   The following are steps taken to deploy model:
   - Prepare a Lambda Function to expose model for inference
   - Package Lambad Function and scripts into a Docker Image
   - Ship Image to AWS ECR
   - Create a Lambda Function on AWS, connect to registered Image on ECR and expose inference function for externeal API access using Function URL


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



