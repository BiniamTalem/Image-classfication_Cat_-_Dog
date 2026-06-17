## Introduction
This project builds upon the work completed in the first semester, where a classification model was developed to distinguish between cat and dog images using Flask. Flask served as the backend framework for deploying the machine learning model and creating a web interface for users to upload images and receive classification results.

In the current semester, I have transitioned this project to use FastAPI instead of Flask. FastAPI is a modern, fast (high-performance) web framework for building APIs with Python. It offers several advantages over Flask, such as automatic generation of interactive API documentation, better performance, and easier handling of asynchronous operations.

By migrating to FastAPI, the project now provides a more efficient and scalable way to serve the image classification model. Users can upload images through an API endpoint, and the server processes the images to classify whether they contain a cat or a dog. The GitHub link included in the response contains all the project files, code, and documentation for this implementation.

This transition enhances the overall functionality and performance of the project, making it more suitable for real-world applications where speed and scalability are important.
# 🐱🐶 Cat-Dog Classification API

The Cat-Dog classification API is a practical demonstration of deploying a deep learning model as a web service. Its primary use is to automaticlly identify whether an uploaded image contains a cat or dog, returning the predicaiton along with a confidence score and per-class probabilities.

## Overview

This project provides a RESTful API service that leverages a trained TensorFlow/Keras deep learning model to classify images of cats and dogs. The API is built with FastAPI for high performance, automatic OpenAPI documentation, and easy integration with other services.


## Practical Applications
- **Pet photo managment**- Automatically tag and organize large collections of pet images in apps or cloud storage.
- **Social media filters**- Add fun overlayes or auto-caption photos based on the detected animal.
- **Veterinary or shelter tools**- Quickly sort incoming images of animals for record-keeping or triage.
- **Smart camera systems**- Trigger actions (e.g., open a door, sound a notification) when a specific pet is detected.
- **E-commerce**- Automatically categorize pet-related product images for better search and recommendations.



## Tech Stack

### Core Technologies
- **Python 3.8+**: Programming language
- **FastAPI 0.115.0**: Web framework
- **TensorFlow 2.15.0**: Deep learning framework
- **Pillow 10.0.0**: Image processing library
- **Uvicorn 0.20.0**: ASGI server

## Project Structure
cat-dog-classification-api/
│
├── app/
│ ├── init.py
│ ├── main.py # FastAPI application entry point
│ ├── schemas.py # Pydantic schemas/models
│ └── model_loader.py # TensorFlow model wrapper
│
├── uploads/ # Uploaded images directory (optional)
│
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .env.example # Environment variables template
