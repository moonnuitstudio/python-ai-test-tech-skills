import os
import spacy
import random
from statistics import mean
from spacy.training.example import Example
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

# SEED
SEED = 45

# Remove overlapping
def remove_overlapping_spans(spans):
    # SORT the spans DESC
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))

    result = []
    last_end = -1

    for start, end, label in spans:
        if start >= last_end:
            result.append((start, end, label))
            last_end = end

    return result

# Converts entities into a set of tuples (start, end)
def to_span_set(entities):
    return set((start, end) for start, end, label in entities)

# MODEL -----------------------------
class Model:
    def __init__(self, model_name="ner_finder_skill_model"):
        self.nlp = spacy.blank("en")
        self.model_name = model_name
        self.ner = None

        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")

    def load_trained_model(self):
        self.nlp = spacy.load(f"data/model/{self.model_name}")

    # Check if the training model exist
    def training_model_exist(self):
        return os.path.isdir(f"data/model/{self.model_name}")

    # TRAINING
    def training(self, training_data, n_iter=20, dropout=0.3):

        # Add LABELS to NER Model
        for _, annotations in training_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        # Get Other pipes to deactivate
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]

        # We only want to train the ner model
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training() # START the training

            print("[*] Starting to train the model...")

            # Start processing
            for itn in range(n_iter):
                print(f"    - Iteration: {itn + 1}")

                random.shuffle(training_data) # Shuffle the data
                losses = {}

                for text, annotations in training_data:
                    doc = self.nlp.make_doc(text)

                    cleaned_entities = remove_overlapping_spans(annotations["entities"])
                    example = Example.from_dict(doc, {"entities": cleaned_entities})

                    # Has been updated to use tunning
                    self.nlp.update([example], drop=dropout, losses=losses)

                print(f"    [!] Loss: {losses['ner']:.4f}")

    # Evaluate the model
    def evaluation(self, evaluation_data):
        y_true = []
        y_pred = []

        for text, annot in evaluation_data:
            doc = self.nlp(text)

            # Prepare SPANS
            pred_spans = set((ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "SKILL")
            true_spans = to_span_set(annot["entities"])

            # binary indicators
            for span in true_spans:
                y_true.append(1)
                y_pred.append(1 if span in pred_spans else 0)

            for span in pred_spans:
                if span not in true_spans:
                    y_true.append(0)
                    y_pred.append(1)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return precision, recall, f1

    # Performs cross-validation and tuning
    @staticmethod
    def cross_validate_and_tuning(training_data, k=5):
        # TUNNING
        param_grid = {
            "n_iter": [10, 20, 25],
            "dropout": [0.2, 0.3]
        }

        # We have to check which one is the best option
        best_score = 0.0
        best_params = {}
        best_model = None

        random.seed(SEED)
        random.shuffle(training_data) # Shuffle Data

        print("[*] CROSS VALIDATION AND TUNING")

        # APPLY TUNNING
        for n_iter in param_grid["n_iter"]:
            for dropout in param_grid["dropout"]:
                print(f"\n[*] Testing combination: n_iter={n_iter}, dropout={dropout}")

                scores = []
                kf = KFold(n_splits=k, shuffle=True, random_state=42)

                fold_model = None # We are going to save the best model

                for fold, (train_idx, test_idx) in enumerate(kf.split(training_data), 1):
                    fold_model = Model()
                    print(f"  - Fold {fold}")

                    train_data = [training_data[i] for i in train_idx]
                    test_data = [training_data[i] for i in test_idx]

                    # TRAINING the model
                    fold_model.training(train_data, n_iter=n_iter, dropout=dropout)

                    precision, recall, f1 = fold_model.evaluation(test_data)
                    scores.append(f1) # SCORES

                # Show the average score
                avg_f1 = mean(scores)
                print(f"  - Average F1: {avg_f1:.4f}")

                # WE use this to check which model has the best result
                if avg_f1 > best_score:
                    best_score = avg_f1
                    best_params = {"n_iter": n_iter, "dropout": dropout}
                    best_model = fold_model

        # DEBUG
        print("\n[!] Best combination found:")
        print(f"   - n_iter:  {best_params['n_iter']}")
        print(f"   - dropout: {best_params['dropout']}")
        print(f"   - F1-Score: {best_score:.4f}")

        return best_model

    # Save the best model
    def save_model(self):
        # SAVE THE MODEL
        self.nlp.to_disk(f"./data/model/{self.model_name}")
        print(f"\n[*] Model saved in data/model/'{self.model_name}'")

    def extract_skills(self, text):
        doc = self.nlp(text)
        return set(ent.text.strip().lower() for ent in doc.ents if ent.label_ == "SKILL")
# -----------------------------------