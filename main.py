import re
import os
import glob
import pandas as pd
import json
from pathlib import Path
from rapidfuzz import fuzz, process
from model import Model
from skillsqlite import DataBaseStore

from jobskillsextractor import prepare_training_data, get_lines_from_description, check_skill

# SET teh dataset file path
DATASET_FILE_PATH = os.path.join(os.path.dirname(__file__), "./data/postings.csv")

# CONSTS -----------------------------------
VALID_TITLES = [ #CONST RELATED TO TECH
    "software engineer", "backend developer", "frontend developer", "full stack developer",
    "web developer", "mobile developer", "react developer", "node.js developer",
    "data scientist", "data analyst", "machine learning engineer", "ai engineer",
    "devops engineer", "cloud engineer", "site reliability engineer", "systems administrator",
    "qa engineer", "test engineer",
    "graphic designer", "ux designer", "ui designer", "product designer",
    "it support", "technical support", "help desk technician",
    "marketing specialist", "digital marketing specialist", "marketing analyst",
    "marketing coordinator", "seo specialist", "ppc specialist", "email marketing specialist",
    "content strategist", "marketing technologist", "performance marketing manager"
]

# This will be used to model the tech titles
COMMON_TITLES = {
    'data analyst', 'data scientist', 'software engineer', 'frontend developer',
    'backend developer', 'full stack developer', 'devops engineer', 'machine learning engineer',
    'ai engineer', 'data engineer', 'cloud architect', 'product manager',
    'software architect', 'systems analyst', 'qa engineer', 'test engineer',
    'network engineer', 'it support specialist', 'mobile developer', 'security analyst',
    'site reliability engineer', 'platform engineer', 'blockchain developer', 'computer vision engineer',
    'nlp engineer', 'mlops engineer', 'data architect', 'business intelligence analyst',
    'cloud engineer', 'data visualization specialist', 'ux designer', 'ui developer',
    'scrum master', 'technical project manager', 'technical writer', 'penetration tester',
    'cybersecurity engineer', 'robotics engineer', 'iot engineer', 'game developer',
    'web developer', 'database administrator', 'data quality analyst', 'bioinformatics engineer',
    'information systems manager', 'cloud security engineer'
}

# USE TO CLEAN CITY FIELD
CITY_COMMON_SUFFIXES = [
    r"\bcentro\b", r"\bdowntown\b", r"\barea\b", r"\bregion\b", r"\bmetropolitan\b",
    r"\bcity\b", r"\bzone\b", r"\bdistrict\b"
]
CITY_COMMON_PREFIXES = [
    r"\bgreater\b", r"\bmetro\b", r"\bmetropolitan\b", r"\barea of\b", r"\bregion of\b"
]

# ------------------------------------------

# Normilize DATA -------------------------------------------------------------------
def clean_title(_title):
    title = _title.lower()
    # Clean prefixes like #123
    title = re.sub(r'^[#\d\-\$\*\!\@\(\)]+', '', title.lower()).strip()
    # ONLY letter and spaces
    title = re.sub(r'[^a-z\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    title = re.sub(r'#\d+\s*', '', title)
    title = re.sub(r'"', '', title)

    # REMOVE NOISE DATA
    title = re.sub(r'visit us.*', '', title)
    title = re.sub(r'opportunity.*', '', title)
    title = re.sub(r' in .*', '', title)
    title = re.sub(r' at .*', '', title)
    title = re.sub(r' full-time.*', '', title)
    title = re.sub(r' part-time.*', '', title)
    title = re.sub(r' temporary.*', '', title)
    title = re.sub(r' intern.*', '', title)

    return title.strip()

def get_new_title(raw_title):
    best_match, score, _ = process.extractOne(raw_title, COMMON_TITLES)

    if score > 80:
        return best_match

    return "other" # OTHER JOBS

# Get the cleanest version of the title
def generalize_title(title):
    _title = clean_title(title)
    return get_new_title(_title)

# Get the cleanest version of the city
def normalize_city(city):
    city = city.lower().strip()

    # DELETE PREFIX
    for prefix in CITY_COMMON_PREFIXES:
        city = re.sub(rf"^{prefix}\s*", "", city)

    # DELETE SUFIX
    for suffix in CITY_COMMON_SUFFIXES:
        city = re.sub(rf"\s*{suffix}$", "", city)

    # CLEAN
    city = re.sub(r"\s+", " ", city)

    return city.title()
# ----------------------------------------------------------------------------------

# FILTERS --------------------------------------------------------------------------
def is_tech_title(title, threshold=85):
    title = title.lower()
    return any(fuzz.partial_ratio(title, valid) >= threshold for valid in VALID_TITLES)
# ----------------------------------------------------------------------------------

# GET data file --------------------------------------------------------------------
def load_training_data():
    train_data = []

    for path in Path("./data/training").glob("train_data_*.json"):
        with open(path, "r", encoding="utf-8") as f:
            train_data.extend(json.load(f))

    return train_data

def get_evaluation_data():
    evaluation_data = None

    with open("./data/evaluation/ner_evaluation_data.json", "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)

    return evaluation_data
# ----------------------------------------------------------------------------------

# PREPARING THE DATA SET -----------------------------------------------------------
def prepare_dataset(df):
    # DROP empy columns
    _df = df.dropna(subset=['description', 'title', 'location'])

    # Get only the city from the location column: San Jose, CA -> San Jose
    _df['city'] = _df['location'].apply(
        lambda x: x.split(',')[0].strip().lower() if isinstance(x, str) and ',' in x else str(x).strip().lower()
    )

    # Clean Title
    _df['title'] = _df['title'].astype(str).apply(generalize_title)

    # REMOVE duplicate data
    return _df.drop_duplicates(subset=['title', 'city', 'description'])
# ----------------------------------------------------------------------------------

# INPUT ----------------------------------------------------------------------------
def check_yes_no_input(msg):
    while True:
        print(f"\n{msg}")
        user_in = input("Please type yes or no: ")

        if not user_in or user_in.lower() not in ("yes", "no", "y", "n"):
            print("[Err] No valid input")
            continue

        break

    return user_in.lower() in ("yes", "y") # RETURN TRUE if we have selected yes
# ----------------------------------------------------------------------------------

# MAIN
def main():
    print("[*]=========================================[*]")
    print("[ ]                                         [ ]")
    print("[ ]                WELCOME                  [ ]")
    print("[ ]                                         [ ]")
    print("[*]=========================================[*]\n")

    print("[*] Preparing dataset...")

    training_files = glob.glob("./data/training/train_data_*.json")

    print("[*] Checking training  files...\n")

    # We read the dataset
    df = pd.read_csv(DATASET_FILE_PATH)

    #FILTER THE DATA SET
    df["title"] = df["title"].astype(str) # ONLY STRING
    df = df[df["title"].apply(is_tech_title)]

    if not training_files:
        print("[!] No training data found...")
        print("[?] Would you like to generate those files?")

        user_in = ""

        # Ask if we want to generate those files
        while True:
            user_in = input("Please type yes or no: ")

            if not user_in or user_in.lower() not in ("yes", "no", "y", "n"):
                print("[Err] No valid input")
                continue

            break

        if user_in.lower() in ("yes", "y"):
            # We take the first 500 rows to prepate training files
            prepare_training_data(df, row_limit=500, offset=0)
            print("\n[!] Please check that the training data is adequate")

        print("GODBYE!...")
    else:
        model = Model() # Prepare the model

        # CHECK if we have the model
        if not model.training_model_exist():
            # DEBUG
            print("[!] Training model does not exist...\n")
            print("[*] Training data has been found...")

            # LOAD TRAINING REGISTER
            training_data = load_training_data()

            print("[*] Training Data Total of Row:", len(training_data))
            print("")

            # CROSS VALIDATION AND TUNING
            model = Model.cross_validate_and_tuning(training_data)

            # Save the best model
            model.save_model()
        else:
            print("[*] Recovering trained model...\n")
            model.load_trained_model()

        # --------------------------------------------------------------------
        print("[*] Loading evaluation data...")

        # Loading test data
        evaluation_data = get_evaluation_data()

        # Preparing Database
        print("[*] Preparing Data Base")
        db = DataBaseStore()

        # SAVE THE SKILLS DATA ----------
        if not db.exists(): # IN CASE WE NEED TO SAVE THE SKILLS
            dataset = prepare_dataset(df)
            dataset_size = len(dataset)

            print("\n[*] DataSet Size:", dataset_size, "rows")

            # Get the every row from the dataset
            inserted = 0
            for i, (_, row) in enumerate(dataset.iterrows(), 1):
                title = str(row.get("title", ""))
                description = str(row.get("description", ""))
                city = str(row.get("city", "")).strip().lower()

                #Check if we have all data so we can start getting the information
                if title == "" or description == "" or city == "":
                    continue

                # Prepate the city
                city = normalize_city(city)

                # GET LINES
                lines = get_lines_from_description(description)

                # DEBUG
                print("\n[+] title:", title)
                print("[+] city:", city)

                inserted = 0

                for line in lines:  # GET EVERY LINE FROM THE DESCRIPTION

                    # IF THERE NO LINES
                    if not line:
                        continue # WE PASS TO THE NEXT ONE

                    skills = model.extract_skills(line) # CHECK THE SKILLS FROM THE LINE

                    # USED to check if we have saved the same skills
                    seen_skills = set()

                    for skill in skills:
                        skill_lower = skill.lower()

                        # POSIble false positive
                        if skill_lower in {"impact", "access", "player"}:
                            continue

                        if skill_lower in seen_skills:
                            continue

                        if check_skill(skill_lower): # CHECK if this is an actuall skill
                            print(f"[-]    - {skill_lower}")
                            db.insert_update(skill_lower, city, title)
                            seen_skills.add(skill_lower)
                            inserted += 1

                # Show the progress if we inserted data
                if inserted > 0:
                    print(f"[+] Skills Inserted: {inserted}")

                # DEBUG
                print(f"[*] Job Done : {i}/{dataset_size}")

        # --------------------------------

        while True:
            print("\nChoose a menu option:")
            print("\t 1. Evaluate current model")
            print("\t 2. Review job titles saved in the system")
            print("\t 3. Review cities saved in the system")
            print("\t 4. Get the 10 most important skills by title and city")

            print("\t 5. Exit\n")

            # GET MENU INPUT OPTION
            user_input = input("Enter the selected menu item: ")

            # Check selected option
            if user_input in ("1", "2", "3", "4", "5"):
                if user_input == "5":
                    break

                match user_input:
                    case "1":
                        # Check if we have information
                        if evaluation_data:
                            precision, recall, f1 = model.evaluation(evaluation_data)

                            print(f"\nEvaluation of the NER model:")
                            print(f"Precision: {precision:.2f}")
                            print(f"Recall:    {recall:.2f}")
                            print(f"F1-Score:  {f1:.2f}")
                            print("")

                    case "2":
                        titles = db.get_all_titles() # Get all titles saved in the system

                        if not titles or len(titles) == 0:
                            print("[ERR] No titles found...")
                            continue

                        max_titles = len(titles) #Get the lenght limit
                        current = 0
                        size = 10

                        print(f"\n[*] List of titles:")
                        print("[*] Total of titles:", max_titles)
                        print("")

                        # Get every 10 titles
                        while current < max_titles:
                            offset = current + size
                            for _t in titles[current:offset]:
                                print(f"[*]  - {_t}")

                            if check_yes_no_input("Would you like to continue?"):
                                current += size
                                print("")
                            else:
                                break

                    case "3":
                        cities = db.get_all_cities()  # Get all cities saved in the system

                        if not cities or len(cities) == 0:
                            print("[ERR] No cities found...")
                            continue

                        max_cities = len(cities)  # Get the lenght limit
                        current = 0
                        size = 10

                        print(f"\n[*] List of cities:")
                        print("[*] Total of cities:", max_cities)
                        print("")

                        # Get every 10 titles
                        while current < max_cities:
                            offset = current + size
                            for _t in cities[current:offset]:
                                print(f"[*]  - {_t}")

                            if check_yes_no_input("Would you like to continue?"):
                                current += size
                                print("")
                            else:
                                break

                    case "4":
                        print("\nEnter job title:")
                        title = input("(software engineer, developer, data analyst, ux designer): ")

                        print("\nEnter city:")
                        city = input("(san jose, sacramento, san francisco, marina): ")

                        skils = db.get_skills(title, city)

                        print(f"\n[*] List of skills:")
                        print("[*] Total of skills:", len(skils))
                        print("")

                        # Show skills list
                        for skill in skils:
                            print(f"[*]  - {skill}")

                        print("\nPress any key to go back to the menu:")
                        input("")




            else:
                print("\n[Err] Invalid input")


        db.close() # CLOSE connection
        print("GODBYE!")

# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()