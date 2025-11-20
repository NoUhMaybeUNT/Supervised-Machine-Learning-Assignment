Decision Tree Solution — UNT Linux Cell Machines (I Generated these instructions with AI, but did verify them)

This solution builds the Figure 7.1 dataset, trains a decision tree on e1–e18, and predicts e19 and e20. It also prints a text view of the tree and saves a PNG visualization.

One-time setup (create a venv in the repo)
Run these commands from the repository root (Supervised-Machine-Learning-Assignment):

- python3 --version
- python3 -m venv .venv
- source .venv/bin/activate
- python -m pip install --upgrade pip
- pip install -r requirements.txt

Run the solution
From the same repository root with the venv active:

- python src/decision_tree_solution.py

Notes
- If your local path to the repo contains spaces (e.g., inside "Programming Assignments/Assignment4"), quote the path when cd'ing into it. After cd'ing into Supervised-Machine-Learning-Assignment, the commands above work as-is.
- Reuse this .venv for subsequent runs. To leave the virtual environment: deactivate
- The outputs/ folder is gitignored; artifacts are generated when you run the program.

What you’ll see
- A preview of the dataset (first 10 rows)
- Predictions for e19 and e20 and the actual labels (0=skips, 1=reads)
- A text dump of the decision tree for interpretability
- A feature-importance listing
- A saved image outputs/decision_tree.png inside the repo
 - Decision paths printed for e19 and e20 showing the exact rules taken
 - A saved CSV outputs/dataset_figure7_1.csv (structured representation)
 - An Evaluation & Discussion section covering interpretability, decision boundaries, and limitations

Notes for UNT Linux cell machines
- Python 3 is available as python3. If python3 isn’t found, try python --version. If needed, open a new shell.
- You can reuse this repo-local .venv for subsequent runs, or alternatively activate your course-level .venv before installing requirements.
- Remember to quote paths that contain spaces, as shown above.
- To leave the virtual environment: run deactivate.

Console verification
On the cell machine, the following worked from the repo root (exit code 0):
- Creating/using .venv at Supervised-Machine-Learning-Assignment/.venv
- Installing requirements from requirements.txt
- Running the script with: python src/decision_tree_solution.py (outputs stored in outputs/)
