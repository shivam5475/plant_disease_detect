This project is a Streamlit application that uses a trained TensorFlow model to recognize plant diseases from uploaded images. It also provides information about the detected diseases using a ChatGPT-powered bot.
Features

Upload plant images for disease detection
Predict plant diseases using a pre-trained TensorFlow model
Provide detailed information about detected diseases using a language model

Installation

Clone the repository:
Copygit clone (https://github.com/shivam5475/plant_disease_detect)
cd plant-disease-recognition

Install the required packages:
Copypip install -r requirements.txt

Set up your environment variables:
Create a .env file in the project root and add your Google API key:
CopyAPI_KEY=your_google_api_key_here

Download the pre-trained model:
Make sure you have the trained_plant_disease_model.keras file in the project root directory.

Usage

Run the Streamlit app:
Copystreamlit run app.py

Open your web browser and go to the URL provided by Streamlit (usually http://localhost:8501).
Upload an image of a plant to detect diseases.
Click the "Predict" button to get the disease prediction and additional information.

Project Structure

app.py: Main Streamlit application
response.py: Contains the resp function for generating disease information
vector.py: Sets up the vector database for disease information retrieval
solutions.txt: Contains information about plant diseases (not included in this repository)

Requirements
See requirements.txt for a full list of dependencies.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

TensorFlow for the machine learning framework
Streamlit for the web application framework
LangChain and Google's Generative AI for natural language processing capabilities
