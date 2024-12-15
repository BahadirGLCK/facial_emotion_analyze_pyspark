from setuptools import setup, find_packages

# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="facial_emotion_analyze_pyspark",
    version="0.1.0",
    description="A PySpark-based facial emotion clustering project",
    author="Bahadir Golcuk",
    author_email="bahadirgolcuk@gmail.com",
    url="https://github.com/BahadirGLCK/facial_emotion_analyze_pyspark",  # Update with your GitHub repo
    packages=find_packages(where="src"),  # Look for packages in the src directory
    package_dir={"": "src"},  # Root of the packages is src/
    install_requires=parse_requirements("requirements.txt"),  # Read from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)