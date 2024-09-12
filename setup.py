from setuptools import setup, find_packages

setup(
    name="plant-disease-recognition",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "tensorflow",
        "numpy",
        "Pillow",
        "keras",
        "python-dotenv",
        "langchain",
        "langchain-google-genai",
        "google-generativeai",
        "chromadb",
        "langchain_community",
    ],
    entry_points={
        "console_scripts": [
            "plant-disease-app=app:main",
        ],
    },
)
