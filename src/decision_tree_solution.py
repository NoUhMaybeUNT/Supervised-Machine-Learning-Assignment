import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt


def build_dataset():
    # Examples e1..e20 from Figure 7.1
    rows = [
        ["known", "new", "long", "home", "skips"],          # e1
        ["unknown", "new", "short", "work", "reads"],       # e2
        ["unknown", "followup", "long", "work", "skips"],   # e3
        ["known", "followup", "long", "home", "skips"],     # e4
        ["known", "new", "short", "home", "reads"],         # e5
        ["known", "followup", "long", "work", "skips"],     # e6
        ["unknown", "followup", "short", "work", "skips"],  # e7
        ["unknown", "new", "short", "work", "reads"],       # e8
        ["known", "followup", "long", "home", "reads"],     # e9
        ["known", "new", "long", "work", "skips"],          # e10
        ["unknown", "followup", "short", "home", "skips"],  # e11
        ["known", "new", "long", "home", "reads"],          # e12
        ["known", "followup", "short", "home", "reads"],    # e13
        ["known", "new", "short", "work", "reads"],         # e14
        ["known", "new", "short", "home", "reads"],         # e15
        ["known", "followup", "short", "work", "reads"],    # e16
        ["unknown", "new", "short", "home", "reads"],       # e17
        ["unknown", "new", "short", "work", "reads"],       # e18

        # e19 and e20: the dataset completes these with User_action e19=reads, e20=skips
        
        ["unknown", "new", "long", "work", "reads"],        # e19
        ["unknown", "followup", "short", "home", "skips"],  # e20
    ]

    cols = ["Author", "Thread", "Length", "Where_read", "User_action"]
    df = pd.DataFrame(rows, columns=cols)
    return df


def prepare_features(df):
    # Encode categorical predictors using one-hot encoding
    X = pd.get_dummies(df[["Author", "Thread", "Length", "Where_read"]])
    # Encode target as 0/1: skips=0, reads=1
    y = (df["User_action"] == "reads").astype(int)
    return X, y


def train_and_evaluate(df):
    X, y = prepare_features(df)

    # According to the assignment: train on e1..e18 and predict for e19,e20
    X_train = X.iloc[0:18]
    y_train = y.iloc[0:18]
    X_test = X.iloc[18:20]
    y_test = y.iloc[18:20]

    clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("Predictions for e19 and e20 (0=skips, 1=reads):", preds)
    print("Actual labels for e19 and e20:", y_test.values)
    print()

    # Print textual representation of the tree
    feature_names = list(X.columns)
    tree_text = export_text(clf, feature_names=feature_names)
    print("Decision tree (text):\n")
    print(tree_text)

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    print("\nFeature importances (one-hot columns):")
    print(importances[importances > 0])

    # Visualize and save tree
    plt.figure(figsize=(16, 8))
    plot_tree(clf, feature_names=feature_names, class_names=["skips", "reads"], filled=True)
    plt.title("Decision Tree trained on e1..e18")
    plt.tight_layout()
    plt.savefig("decision_tree.png")
    print('\nSaved tree visualization to `decision_tree.png`.')

    # Show decision path for each test example
    print('\nDecision paths for test examples:')
    node_indicator = clf.decision_path(X_test)
    leaf_ids = clf.apply(X_test)
    tree_ = clf.tree_

    for sample_id in range(X_test.shape[0]):
        print(f"\nExample e{19 + sample_id}:")
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
        for node_id in node_index:
            if tree_.feature[node_id] != -2:
                name = feature_names[tree_.feature[node_id]]
                threshold = tree_.threshold[node_id]
                # Get sample's value for that feature
                sample_val = X_test.iloc[sample_id, tree_.feature[node_id]]
                print(f" Node {node_id}: ( {name} <= {threshold:.2f} ) -- sample value = {sample_val}")
            else:
                print(f" Node {node_id}: leaf node.")

    # Evaluation & discussion output to reflect assignment requirements
    print("\n--- Evaluation & Discussion ---")
    print(f"Tree depth: {clf.get_depth()} | Number of leaves: {clf.get_n_leaves()}")

    # Summarize top features
    if (importances > 0).any():
        top_feats = importances[importances > 0].head(5)
        print("Top contributing features (by impurity decrease):")
        for feat, val in top_feats.items():
            print(f" - {feat}: {val:.3f}")
    else:
        print("Feature importance not informative (all zeros).")

    print("\nInterpretability:")
    print(" - The model is a decision tree with axis-aligned splits on one-hot encoded features.")
    print(" - Since one-hot columns are binary, thresholds ~0.5 correspond to rules like 'feature present vs not present'.")

    print("\nDecision boundaries:")
    print(" - Each split partitions the space based on a single encoded category, forming simple if-then rules.")
    print(" - See 'Decision paths' above for explicit rules followed by e19 and e20.")

    # Basic limitations
    reads_count = int(y.iloc[0:18].sum())
    skips_count = int((1 - y.iloc[0:18]).sum())
    print("\nLimitations:")
    print(f" - Small training set (n=18) risks overfitting; class balance reads={reads_count}, skips={skips_count}.")
    print(" - Categorical handling via one-hot increases dimensionality; deeper trees may latch onto noise.")
    print(" - 'Length' is treated as categorical (short/long); no ordinal modeling here.")
    print(" - Contextual factors (e.g., author reputation dynamics, time) are not modeled.")


def main():
    df = build_dataset()
    print("Dataset preview (first 10 rows):\n")
    print(df.head(10))
    # Save a structured representation to CSV to reflect preprocessing requirement
    df.to_csv("dataset_figure7_1.csv", index=False)
    print("\nSaved structured dataset to dataset_figure7_1.csv")
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
