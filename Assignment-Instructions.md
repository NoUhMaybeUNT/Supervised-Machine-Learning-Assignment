# Supervised Machine Learning Assignment — Decision Trees

## Figure 7.1 — Example Data of a User’s Behavior

### Training examples (e1–e18)

| Example | Author  | Thread   | Length | Where_read | User_action |
|:-------:|:-------:|:--------:|:------:|:----------:|:-----------:|
| e1      | known   | new      | long   | home       | skips       |
| e2      | unknown | new      | short  | work       | reads       |
| e3      | unknown | followup | long   | work       | skips       |
| e4      | known   | followup | long   | home       | skips       |
| e5      | known   | new      | short  | home       | reads       |
| e6      | known   | followup | long   | work       | skips       |
| e7      | unknown | followup | short  | work       | skips       |
| e8      | unknown | new      | short  | work       | reads       |
| e9      | known   | followup | long   | home       | reads       |
| e10     | known   | new      | long   | work       | skips       |
| e11     | unknown | followup | short  | home       | skips       |
| e12     | known   | new      | long   | home       | reads       |
| e13     | known   | followup | short  | home       | reads       |
| e14     | known   | new      | short  | work       | reads       |
| e15     | known   | new      | short  | home       | reads       |
| e16     | known   | followup | short  | work       | reads       |
| e17     | unknown | new      | short  | home       | reads       |
| e18     | unknown | new      | short  | work       | reads       |

### New cases (e19–e20)

| Example | Author  | Thread   | Length | Where_read | User_action |
|:-------:|:-------:|:--------:|:------:|:----------:|:-----------:|
| e19     | unknown | new      | long   | work       | ?           |
| e20     | unknown | followup | short  | home       | ?           |

Figure 7.1 shows fictitious examples obtained from observing a user deciding whether to read articles posted to a threaded discussion website, depending on whether the author is known or not, whether the article started a new thread or was a follow-up, the length of the article, and whether it is read at home or at work. `e_1, …, e_18` are the training examples. The aim is to make a prediction for the user action on `e_19`, `e_20`, and other, currently unseen, examples.

## Example 7.1

Figure 7.1 shows training examples typical of a classification task. The aim is to predict whether a person reads an article posted to a threaded discussion website given properties of the article. The input features are Author, Thread, Length, and Where_read. There is one target feature, User_action. The domain of Author is {known, unknown}, the domain of Thread is {new, followup}, and so on.

There are eighteen training examples, each of which has a value for all of the features. In this dataset, `Author(e_11) = unknown`, `Thread(e_11) = followup`, and `User_action(e_11) = skips`.

There are two new cases, `e_19` and `e_20`, for which the model needs to predict the user action.

---

## Problem Overview

Students need to write a computer program to implement a decision tree learner on the dataset shown in Figure 7.1. To complete the dataset, take the user action for examples `e19` and `e20` as reads and skips, respectively. Use any suitable technique to split the dataset into training and testing subsets.

## Tasks

- Data Preprocessing
	- Encode categorical features using one-hot or label encoding.
	- Represent the dataset in a structured format (e.g., CSV, dictionary, or pandas DataFrame).

- Decision Tree Implementation
	- Implement a decision tree classifier from scratch or use a library (e.g., scikit-learn) with clear justification.
	- Train the model on examples `e1` to `e18`.

- Prediction
	- Use the trained model to predict `User_action` for `e19` and `e20`.

- Tree Visualization
	- Display the decision tree structure using text or graphical tools (e.g., graphviz, plot_tree).
	- Highlight decision paths for `e19` and `e20`.

- Evaluation & Discussion
	- Discuss the interpretability of your tree.
	- Analyze feature importance and decision boundaries.
	- Reflect on limitations (e.g., overfitting, small dataset size).