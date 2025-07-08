import math
import os
import json
import time
import re
import spacy
from rapidfuzz import fuzz

# CONSTANTS ------------------------------------
ONET_PATH = os.path.join(os.path.dirname(__file__), "./data/techno_skills.txt")
ONET_SKILLS = set()

# USED TO FILTER NO TECH JOBS
JOB_TITLES = {"manager", "assistant", "coordinator", "supervisor", "chef", "therapist", "developer", "analyst", "engineer", "technician", "officer"}
# ----------------------------------------------

# PREPATE THE NLP Object -----------------------
nlp = spacy.load("en_core_web_sm")
# ----------------------------------------------

# EXTRA SKILLS -----------------------------------------------
CUSTOM_TECH_SKILLS = [
    "a/b testing", "access", "adobe creative cloud", "adobe xd", "after effects", "agile",
    "ahrefs", "alpine.js", "alteryx", "angular", "ansible", "asana", "asp.net",
    "aws", "aws cloudformation", "backbone.js", "bash", "bigquery", "bootstrap",
    "burp suite", "c#", "c++", "call center software", "canva", "case management",
    "cassandra", "chatbots", "ci/cd", "cisco", "cloudwatch", "computer", "confluence",
    "content marketing", "crm", "css", "customer service", "cypress", "dynamodb",
    "digitalocean", "django", "docker", "drift", "elasticsearch", "ember.js", "email support",
    "express", "excel", "facebook ads", "fastapi", "figma", "firebase", "flask",
    "freshdesk", "front-end", "gensim", "gitlab ci", "go", "google ads",
    "google analytics", "google cloud", "google data studio", "grafana", "graphql",
    "help desk", "heroku", "huggingface", "html", "hubspot", "iam", "illustrator",
    "incident management", "indesign", "invision", "itil", "java", "jenkins", "jest",
    "jira", "jquery", "json", "junit", "kotlin", "keras", "knowledge base management",
    "laravel", "lightgbm", "linux", "looker", "mailchimp", "mariadb", "marketing automation",
    "matplotlib", "metabase", "microsoft dynamics", "microsoft excel", "microsoft office",
    "microsoft outlook", "microsoft teams", "microsoft word", "mlflow", "mongodb",
    "monday.com", "mocha", "moz", "mysql", "netlify", "next.js", "nginx", "node.js",
    "notion", "npm", "nuxt.js", "numpy", "oauth", "office 365", "opencv", "onenote",
    "oracle", "outlook", "owl carousel", "owasp", "pandas", "penetration testing",
    "perl", "photoshop", "php", "phone support", "postgreSQL", "postman", "power apps",
    "power automate", "power bi", "powerpoint", "preact", "premiere pro", "prometheus",
    "proficiency", "publisher", "pytorch", "pytest", "python", "r", "react", "redis",
    "redshift", "remote support", "rest api", "ruby", "ruby on rails", "s3",
    "salesforce", "sass", "scikit-learn", "scrum", "scss", "selenium", "sem", "semrush",
    "sentry", "seo", "service now", "sharepoint", "shell scripting", "siem", "sketch",
    "slack", "sla management", "snowflake", "soap", "social media management", "sso",
    "spacy", "spring boot", "sqlite", "sso", "ssl", "svelte", "swift", "tableau",
    "tailwind css", "technical support", "tensorflow", "terraform", "ticketing systems",
    "transformers", "trello", "troubleshooting", "typescript", "unittest", "vue",
    "visio", "voip", "web development programming", "windows server", "wireless networks",
    "word", "zendesk", "zapier", "xml", "xgboost", "yaml", "power point", "powerpoint",
    "wordpress", "elementor", "divi", "html 5", "golang", "figma", "unity", "c#", "c++"
]
# -----------------------------------------------------------

# GET THE DATA FROM TECHNO SKILLS --------------
if os.path.exists(ONET_PATH):
    with open(ONET_PATH, encoding="utf-8") as f:
        # GET THE LINE
        for line in f:
            # Split the content
            parts = line.strip().split('\t')
            # If we have the column skill
            if len(parts) >= 2:
                skill = parts[1].strip().lower()
                if len(skill.split()) >= 2 or len(skill) > 5:
                    ONET_SKILLS.add(skill)

    ONET_SKILLS.update(CUSTOM_TECH_SKILLS)

    print(f"[*] Total of skills found: {len(ONET_SKILLS)}")
# ----------------------------------------------

# CONTEXT --------------------------------------
# LIST of context in tech
CONTEXT_PATTERNS = [
    r"experience with", r"experienced in", r"hands-on experience with",
    r"proficient in", r"proficiency with", r"expert in", r"expertise in",
    r"knowledge of", r"strong knowledge of", r"working knowledge of",
    r"skills in", r"skilled in", r"background in",
    r"used with", r"used for", r"use of", r"familiar with",
    r"working with", r"worked with", r"exposure to",
    r"ability to use", r"capable of using", r"understanding of",
    r"technologies such as", r"familiarity with", r"required skills", r"working in",
    r"responsibilities"
]

# CHECK if it location
NEGATIVE_CONTEXTS = {
    "are located in", "located in", "located at",
    "is based in", "based in", "headquartered in"
}

context_regex = re.compile(rf"(?:{'|'.join(CONTEXT_PATTERNS)})\s(.{{0,100}})", re.IGNORECASE)
# ----------------------------------------------

# Split description field into  multiple lines
def split_text_by_dot_and_newline(text):
    # List of lines
    parts = []

    # Get very line
    for segment in text.split('\n'):
        segment = segment.strip() #Get current element string

        # Check if theres not element
        if not segment:
            continue

        sub_segments = segment.split('.') # We separate the string but this time by dot

        # CHeck every sub string
        for sub in sub_segments:
            sub = sub.strip()

            if sub:
                # Remove the first two character if start with
                if sub.startswith(". "):
                    sub = sub[2:]

                parts.append(sub) # Save the sub string

    return parts

# Find Skill
def find_skill_spans(text, threshold=90):
    spans = [] # Prepare all location
    lowered = text.lower() #Normalize the text
    matched_skills = set()

    # GET ALL CONTEXT
    context_matches = context_regex.findall(lowered)

    # If theres no matches so we return as an empty
    if not context_matches:
        return []

    # Check context
    for ctx in context_matches:
        # In case we face a negative context
        if any(neg in ctx for neg in NEGATIVE_CONTEXTS):
            continue

        #Cleen the context
        ctx_cleaned = re.sub(r'(?<!\s)must', '', ctx.lower())
        ctx_cleaned = re.sub(r'(?<!\s)have', '', ctx_cleaned)

        doc = nlp(ctx_cleaned)

        for i in range(len(doc)):
            for j in range(i + 1, min(i + 6, len(doc)) + 1):
                candidate = doc[i:j].text.strip()

                # CLEAN
                candidate_clean = re.sub(r"[^\w\s\-+]", "", candidate).strip()
                candidate_clean = re.sub(r"^(and|or|with|using|via)\s+", "", candidate_clean)
                candidate_clean = re.sub(r"\s+(and|or|with|using|via)$", "", candidate_clean)

                # Try to normalize the text ---------------------------------------------------------------
                if len(candidate_clean) < 2:
                    continue

                if not re.match(r"^[a-zA-Z0-9\s\-\+]{3,}$", candidate_clean):
                    continue

                if "of " in candidate_clean:
                    continue

                token = nlp(candidate_clean)[0]
                if token.pos_ in {"VERB", "AUX", "DET", "ADJ"}:
                    continue

                if re.match(r"^(to|for|by|with|and)\b", candidate_clean):
                    continue

                last_token = candidate_clean.split()[-1]
                if last_token in {"in", "of", "for", "to", "with", "and", "on"}:
                    continue

                tokens = candidate_clean.lower().split()
                if all(t in JOB_TITLES for t in tokens if len(t) > 2):
                    continue

                if any(candidate_clean in m for m in matched_skills if candidate_clean != m and len(candidate_clean) < len(m)):
                    continue
                # -----------------------------------------------------------------------------------------

                # SPLIT the candidates by and / or
                split_candidates = re.split(r"\band\b|\bor\b|,", candidate_clean)

                #Check every sub candidate
                for sub_candidate in split_candidates:
                    sub_candidate = sub_candidate.strip()

                    if len(sub_candidate) < 2:
                        continue

                    # Fuzzy match
                    for skill in ONET_SKILLS:
                        skill_lower = skill.lower()

                        if fuzz.partial_ratio(skill_lower, sub_candidate.lower()) >= threshold:

                            # Filter: avoid false positives due to truncated words
                            if not re.search(rf"\b{re.escape(skill_lower)}\b", ctx_cleaned):
                                continue

                            try:
                                # First attempt: with word delimiters
                                pattern = re.compile(rf"\b{re.escape(skill_lower)}\b", re.IGNORECASE)
                                matches = list(pattern.finditer(text.lower()))

                                # If there are no matches and it was in the clean context, relax the search
                                if not matches and skill_lower in ctx_cleaned:
                                    # Search without \b, but on the original text
                                    pattern = re.compile(re.escape(skill_lower), re.IGNORECASE)
                                    matches = list(pattern.finditer(text.lower()))

                                for match in matches:
                                    start, end = match.start(), match.end()

                                    if (start, end) not in spans:
                                        spans.append((start, end, "SKILL"))
                                        matched_skills.add(skill_lower)
                            except ValueError:
                                continue

                            break  # Next possible skill

    return spans

# Save the training data in out
def save_training_data(training_data, chunk_size=50, file_base_name="train_data"):
    total = len(training_data)
    num_chunks = math.ceil(total / chunk_size)

    for i in range(num_chunks):
        # Prepare the portion
        start = i * chunk_size
        end = start + chunk_size

        # Get the data
        chunk = training_data[start:end]

        # Set the filename
        filename = f"./out/{file_base_name}_{i+1}.json"

        # Save the file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)

        # DEBUG
        print(f"[✓] {filename} was saved with {len(chunk)} examples")

# Check if skills is valid
def check_skill(skill):
    return skill.lower() in ONET_SKILLS

# Prepare teh texto so we can start checking the information
def get_lines_from_description(description):
    text = description.replace('\r\n', '\n').replace('\r', '\n')

    # GET the part of the text
    return split_text_by_dot_and_newline(text)

# PREPARE THE TRAINING DATA
def prepare_training_data(df, row_limit=500, offset=0):
    row_limit = offset + row_limit

    training_data = []

    for i, row in enumerate(df.iterrows()):
        if i < offset:
            continue

        if i >= row_limit:
            break

        # Main Information
        _title = str(row[1].get("title", "")).strip()
        _description = str(row[1].get("description", "")).strip().lower()

        # Prepare the string to split them
        _description = _description.replace('\r\n', '\n').replace('\r', '\n')

        # GET the part of the text
        fragments = split_text_by_dot_and_newline(_description)

        for line in fragments: # GET EVERY LINE FROM THE DESCRIPTION

            if not line:
                continue

            start_time = time.time()

            spans = find_skill_spans(line) # GET the best options

            # CHECK if we already have reviewed the same skill
            seen = set()
            deduped_spans = []

            # GET the found skills
            for start, end, label in spans:
                skill_text = line[start:end]

                if (
                    skill_text.lower() not in seen
                    and len(skill_text) > 1
                ):
                    deduped_spans.append((start, end, label))
                    seen.add(skill_text.lower())

            # Debug
            was_added = False

            if deduped_spans:
                training_data.append((line, {"entities": deduped_spans}))
                was_added = True

            print(
                f"[{i + 1}/{row_limit}] Job Done {time.time() - start_time:.2f} s -> {'Used' if was_added else 'No Used'}")


    for i, (text, annot) in enumerate(training_data):
        print(f"\n--- Example {i + 1} ---")
        print(text)
        for start, end, label in annot["entities"]:
            print(f"  → {label}: '{text[start:end]}'")
        print("")

    if training_data:
        #Every element will have only 50 registers
        save_training_data(training_data=training_data, chunk_size=50, file_base_name="train_data")

    else:
        print("[!] No valid examples found")