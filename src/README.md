Decision Tree Solution — UNT Linux Cell Machines

This solution builds the Figure 7.1 dataset, trains a decision tree on e1–e18, and predicts e19 and e20. It also prints a text view of the tree and saves a PNG visualization.

These instructions are tailored for the CSE Linux cell machines (home directory under /nfs/home/STUDENTS/...). Paths with spaces are quoted.

One-time setup (course root venv)
Run these commands from your course folder (replace nvs0025 with your EUID):

   cd /nfs/home/STUDENTS/<your-euid>/CSCE4201
   python3 --version
   # Create a virtual environment once at the course root (if it doesn't exist):
   python3 -m venv .venv
   # Activate it for this shell session:
   source .venv/bin/activate
   # Upgrade pip (optional) and install assignment requirements:
   python -m pip install --upgrade pip
   pip install -r "Programming Assignments/Assignment4/Supervised-Machine-Learning-Assignment/requirements.txt"

Run the solution (recommended)
From the same CSCE4201 course folder with the venv active:

   python "Programming Assignments/Assignment4/Supervised-Machine-Learning-Assignment/src/decision_tree_solution.py"

Alternative: run from the assignment folder

   cd \
   "/nfs/home/STUDENTS/<your-euid>/CSCE4201/Programming Assignments/Assignment4/Supervised-Machine-Learning-Assignment"
   # Activate the course venv from two levels up
   source ../../.venv/bin/activate
   # Ensure deps are installed in the venv
   pip install -r requirements.txt
   # Run
   python src/decision_tree_solution.py

What you’ll see
- A preview of the dataset (first 10 rows)
- Predictions for e19 and e20 and the actual labels (0=skips, 1=reads)
- A text dump of the decision tree for interpretability
- A feature-importance listing
- A saved image decision_tree.png in the current working directory
 - Decision paths printed for e19 and e20 showing the exact rules taken
 - A saved CSV dataset_figure7_1.csv (structured representation)
 - An Evaluation & Discussion section covering interpretability, decision boundaries, and limitations

Notes for UNT Linux cell machines
- Python 3 is available as python3. If python3 isn’t found, try python --version. If needed, open a new shell.
- Keep using the same .venv at the CSCE4201 course root for all assignments to save space.
- Remember to quote paths that contain spaces, as shown above.
- To leave the virtual environment: run deactivate.

Console verification
On the cell machine, the following worked (exit code 0):
- Creating/using .venv at /nfs/home/STUDENTS/<your-euid>/CSCE4201/.venv
- Installing requirements from requirements.txt
- Running the script with: python "Programming Assignments/Assignment4/Supervised-Machine-Learning-Assignment/src/decision_tree_solution.py"
