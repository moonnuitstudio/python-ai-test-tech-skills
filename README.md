<strong> **DO NOT DISTRIBUTE OR PUBLICLY POST SOLUTIONS TO THESE LABS. MAKE ALL FORKS OF THIS REPOSITORY WITH SOLUTION CODE PRIVATE. PLEASE REFER TO THE STUDENT CODE OF CONDUCT AND ETHICAL EXPECTATIONS FOR COLLEGE OF INFORMATION TECHNOLOGY STUDENTS FOR SPECIFICS. ** </strong>

# WESTERN GOVERNORS UNIVERSITY

## D683 – ADVANCED AI AND ML

This project is designed to extract technical skills from job descriptions using a combination of Natural Language Processing (NLP) techniques, context-based filtering, and fuzzy matching. The extracted information is then used to generate annotated training data suitable for training Named Entity Recognition (NER) models. These models will later be used to automatically process the remaining job descriptions in the dataset and identify the most relevant technical skills based on the specified job title and city.

### Dataset

The dataset used in this project is from Kaggle: [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings). Which, it contains over 140,000 job postings collected from LinkedIn.

Due to the large volume of data, a filtering step was applied to select only postings where the job title matches roles in the tech industry, such as:

- Full Stack Developer  
- Software Engineer  
- Data Analyst  

From the filtered data, 500 records were selected and processed. Skill extraction is based on:

- A custom manually curated list of technical skills  
- An official list from [onetcenter.org](https://www.onetcenter.org/)  

These examples are used to generate training files in JSON format.

> **Note:**  
> The cleaned and ready-to-use training data is located in the `./data/training` directory.  
> If these files are missing, the program will prompt you to generate them. In that case, it will automatically create new training samples and store them in the `./out` directory.  
>  
> This separation ensures that all generated files can be reviewed and verified before being finalized for training use.

---

## Requirements

To run this project successfully, ensure you have the following software and packages installed.

### Software Requirements
- Python 3.10 or higher
- Git (optional, for cloning the repository)
- SQLite (included with Python’s standard library via `sqlite3`)

### Hardware Requirements
- Minimum **4 GB RAM**  
- Recommended **8 GB RAM** for efficient model training
- At least **600 MB** of free disk space to store:
  - NLP models
  - Training data
  - SQLite database (`aiml.sqlite`)
  - Output reports or skill indexes

### Development & Debugging
- **Jupyter Notebook** – for interactive testing and data inspection
- **VSCode** or **PyCharm** – recommended for better Python project management
- **Git** – to version control your model, scripts, and training iterations

---

## AI-Powered Skill Extraction System

This Python application is designed to extract technical skills from job descriptions using Natural Language Processing (NLP) and Named Entity Recognition (NER). It automates the annotation, training, extraction, and storage of skills relevant to each job posting by city and title.

### Create a Virtual Environment

```bash
python -m venv env
```

### Activate the environment

- On Windows:

```cmd
.\env\Scripts\activate
```

- On Linux/macOS:

```bash
source env/bin/activate
```

### Install required packages

```bash
pip install -r requirements.txt   
```

### Run the program

To start the system, execute:

```bash
python main.py   
```

---

## The Application Performs the Following:

1. **Load Dataset and Skill List**
   - Retrieves job postings from the dataset.
   - Loads technical skills from [O*NET Technology Skills](https://www.onetcenter.org/dl_files/database/db_23_2_text/Technology%20Skills.txt).
   - Prepares the data for skill recognition.

  
2. **Check or Generate Training Data**
   - Searches for existing training examples in `/data/training`.
   - If not found, generates annotated training samples automatically and saves them to `/out`.
   - These generated samples **must be reviewed and copied manually** to `/data/training` before training.

  
3. **Train or Load the NER Model**
   - If training data exists, the model is trained using cross-validation and hyperparameter tuning.
   - If a trained model already exists, it will be loaded automatically from `data/model/`.

  
4. **Extract Skills from Descriptions**
   - The trained model processes the job descriptions.
   - It identifies and extracts skills based on the context of the text.
   - Normalizes job titles and city names.
  

5. **Store Results in Database**
   - Saves extracted skills and metadata (title, city) into an SQLite database (`aiml.sqlite`).
   - This enables fast querying without re-processing large datasets.
  

6. **Command-Line Interface (CLI) Menu**
   - After setup, a CLI appears offering five options:
     - View model metrics.
     - List all job titles.
     - List all cities.
     - Query top 10 skills by job title and city.
     - Exit the program.

