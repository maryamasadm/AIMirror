# AI Mirror
This app analyses your visual first impression for sample images that simulate a videocall situation. 

![mirror picture](https://github.com/maryamasadm/AIMirror/main/0029.jpg)
*Caption: Look at AL Mirror ! *

## Online App 

The app is available at https://huggingface.co/spaces/miloszpaul/aimirror

## Local Installation

### Setting up the Conda environment

To set up the Conda environment needed to run the MeetPersona project locally on your machine, follow these steps:

1. Ensure that you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

2. Clone the MeetPersona repository to your local machine: git clone https://github.com/maryamasadm/AiMirror.git

3. Create the Conda environment from the `environment.yml` file: conda env create -f environment.yml

3. Activate the newly created environment: conda activate unified_env

4. To start the Streamlit application, run: streamlit run app.py

**Acknowledgments:**

- [Maryam](www.linkedin.com/in/maryamasadzadeh): Developed computer vision and AI model codes.
- [Milosz](https://github.com/miloszpaul): Deployment and research.
- [Marzieh](https://www.linkedin.com/in/marzieh-goljahi-150b39265/): Developed AI code and research.
- [Dr.Shivakumara](https://www.linkedin.com/in/palaiahnakote-shivakumara-8b23a215/): Provided guidance and valuable input on project ideas and dataset selection.

Special thanks to everyone involved in making this project a success.

**Libraries and Frameworks:**

- [YOLOv8](https://docs.ultralytics.com/): Used for object detection.
- [ResNet50 (Fastai)](https://github.com/fastai/fastai): Utilized ResNet50 architecture for regression.

