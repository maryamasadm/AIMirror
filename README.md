# MeetPersona
This app analyses your visual first impression for sample images that simulate a videocall situation. 

## Online App 

The app is available at https://meetpersona.streamlit.app/

## Local Installation

### Setting up the Conda environment

To set up the Conda environment needed to run the MeetPersona project locally on your machine, follow these steps:

1. Ensure that you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

2. Clone the MeetPersona repository to your local machine: git clone https://github.com/miloszpaul/MeetPersona.git

3. Create the Conda environment from the `environment.yml` file: conda env create -f environment.yml

3. Activate the newly created environment: conda activate unified_env

4. To start the Streamlit application, run: streamlit run app.py
