from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing  import OneHotEncoder

input_file = "2_digital_library_items_noisy_high_access.csv"
test_size = 0.3
random_state = 42
target_col = 'high_access'

numerical_features = [
    'file_size_mb',
    'num_pages_or_frames',
    'content_length_tokens',
    'access_count_30d',
    'duration_seconds',
    'upload_year',
    'upload_day_of_year',
    'num_detected_objects',
]

remove_columns = [
    'item_id',
    'collection_id',
    'subject_tags',
]

def train_and_eval_model(x, y, cv_splitter, model_name="model"):

    # x.to_csv("xme.csv", index=False)
    # y.to_csv("yme.csv", index=False)

    # Split data to a train and test set
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Supervised learning model - Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    # Evaluate model
    accuracy_score_value = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # Cross-validation
    cv_scores = cross_val_score(clf, x, y, cv=cv_splitter, scoring='accuracy')

    return {
        "model_name": model_name,
        "accuracy": accuracy_score_value,
        "confusion_matrix": cm,
        "classification_report": report,
        "cv_mean": cv_scores.mean(),
    }


def main():

    df = pd.read_csv(input_file)
    print(f"Loaded data with shape: {df.shape}")

    # DATA CLEANING

    df = df.drop(columns=remove_columns)

    # FEATURE ENGINEERING

    cat_features = [
        col for col in df.columns
        if col not in numerical_features + remove_columns + [target_col]
    ]

    # Separate features and target
    y = df[target_col].astype(int)

    # Handle categorical features with One-Hot Encoding
    if cat_features:
        print("Categorical features detected:", cat_features)

        ohe = OneHotEncoder(sparse_output=False, dtype='float', handle_unknown='ignore')
        cat_matrix = ohe.fit_transform(df[cat_features])

        ohe_feature_names = ohe.get_feature_names_out(cat_features)
        print("One-Hot Encoded feature names:", ohe_feature_names)

        cat_df = pd.DataFrame(cat_matrix, columns=ohe_feature_names, index=df.index)
    else:
        cat_df = pd.DataFrame(index=df.index)

    x = pd.concat([df[numerical_features], cat_df], axis=1)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    # BASELINE MODEL with all features
    baseline_model = train_and_eval_model(x, y, cv, model_name="baseline")

    print ("BASELINE MODEL RESULTS with all features")
    print (f"Accuracy: {baseline_model['accuracy']}")
    print (baseline_model["confusion_matrix"])
    print (baseline_model["classification_report"])

    # Feature Selection using Pearson product-moment correlation
    # coefficients between variables
    correlations = {}
    for col in x.columns:
        if x[col].std() == 0:
            corr_val = 0
        else:
            corr_val = np.corrcoef(x[col], y)[0, 1]
        correlations[col] = abs(corr_val)
    ranked_corr = sorted(correlations.items(), key=lambda v: v[1], reverse=True)
    print(f"Ranked feature correlations: {ranked_corr}")

    # MODEL with top 3 correlated features
    top_features = [v[0] for v in ranked_corr[:3]]
    x_corr = x[top_features]

    top3_model = train_and_eval_model(x_corr, y, cv, model_name="baseline")

    print ("MODEL RESULTS - top 3 correlated features")
    print(f"Features: {top_features}")
    print (f"Accuracy: {top3_model['accuracy']}")
    print (top3_model["confusion_matrix"])
    print (top3_model["classification_report"])

    # MODEL with best accuracy metric over all
    # combinations of features

    best_model = {"accuracy": 0.0}
    best_features = None

    all_corr = [v[0] for v in ranked_corr[:10]]
    for r in range(3, len(all_corr) + 1):
        print(f"{r=}")

        for features in combinations(all_corr, r):
            x_corr = x[list(features)]

            test_model = train_and_eval_model(x_corr, y, cv, model_name="baseline")

            if test_model["accuracy"] > best_model['accuracy']:
                best_model = test_model
                best_features = features

    print ("MODEL RESULTS - best accuracy over combinations of top 10 correlated features")
    print(f"Features: {best_features}")
    print (f"Accuracy: {best_model['accuracy']}")
    print (best_model["confusion_matrix"])
    print (best_model["classification_report"])


if __name__ == "__main__":
    main()
