Solution runner and instructions

Run the solution which builds the dataset (Figure 7.1), trains a decision tree on examples e1..e18, and predicts e19 and e20.

Quick commands (from the repository root):

1) Install dependencies into the configured environment:

   .../.venv/bin/python -m pip install --user -r requirements.txt

2) Run the script:

   .../.venv/bin/python src/decision_tree_solution.py

Outputs:
- Prints predictions for e19 and e20 and the actual labels
- Saves a tree visualization image `decision_tree.png` in the working directory

Notes:
- The script uses one-hot encoding and a scikit-learn DecisionTreeClassifier (entropy criterion) for interpretability.
- The provided `requirements.txt` lists required packages.
