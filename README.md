<div align='center'>
<h1>Machine-Learning-Code-Submission-Analyzer</h1>
<p>A MATLAB ML model that predicts Codeforces submission outcomes.</p>
</div>
<br>

## Overview
This MATLAB-based machine learning project classifies Codeforces submissions into three categories:

- **Accepted** (Correct Solutions)
- **Bug** (Incorrect Solutions: WA, RTE)
- **Inefficiency** (Performance Issues: TLE, MLE)

Using features like execution time, memory usage, and problem rating, it trains multiple classifiers—Decision Tree, Random Forest, KNN, and SVM—to predict whether a new submission is likely to be correct, buggy, or inefficient. The project utilizes cross-validation, feature scaling, and confusion matrices for evaluation, making it a powerful tool for analyzing competitive programming performance trends.

## Workflow

### 1. Data Fetching
Retrieves your Codeforces submission data using the Codeforces API.

### 2. Feature Extraction
Extracts several numerical features from each submission:

- `PassedTestCount`: Number of test cases passed.
- `TimeConsumedMillis`: Execution time in milliseconds.
- `MemoryConsumedBytes`: Memory usage in bytes.
- `RelativeTimeSeconds`: Time relative to the contest start.
- `ProblemRating`: Difficulty rating of the problem (if available or simulated).

### 3. Labeling
Each submission is labeled into one of three categories:

- **Accepted** (0)
- **Bug** (1)
- **Inefficiency** (2)

### 4. Data Balancing
The dataset is balanced across the three classes if possible.

### 5. Model Training & Evaluation
Four classifiers are trained using 5-fold cross-validation (or a hold-out split if data is limited):

- **Decision Tree** (with pruning)
- **Random Forest** (bagging with 150 cycles)
- **K-Nearest Neighbors** (KNN with 7 neighbors)
- **Support Vector Machine** (SVM, implemented via `fitcecoc` for multi-class classification)

Feature scaling (z-score normalization) is applied within each fold.

### 6. Evaluation & Visualization
Aggregated confusion matrices are computed over all folds, and several figures are generated to visualize:

- **Scatter plot** of Passed Test Count vs. Time Consumed.
- **Histogram** of Problem Ratings.
- **Confusion charts** for each classifier.

## Installation & Usage

### Open MATLAB Online:
- Log in to your MATLAB Online account.

### Create & Save the Script:
- Create a new script and paste the content from `CodeforcesSubmissionClassifier.m`.
- Save the file with a `.m` extension.

### Clear Workspace:
Run the following in the Command Window:
```matlab
clear; clc;
```

### Run the Script:
Click the **Run** button in the Editor or type the script name in the Command Window:
```matlab
CodeforcesSubmissionClassifier
```

### View Results & Figures:
- The script will display average accuracy for each classifier in the Command Window.
- Figures will be generated for:
  - **Scatter Plot of Passed Test Count vs. Time Consumed.**
  - **Histogram of Problem Ratings.**
  - **Aggregated Confusion Matrices for Decision Tree, Random Forest, KNN, and SVM.**

## Visualizations

### Figure 1: Scatter Plot
Passed Test Count vs. Time Consumed (ms) Colored by Submission Category.

<img src="https://github.com/user-attachments/assets/3aa34f27-2749-4c87-b681-bc9630407326" width="400"/>

### Figure 2: Histogram
Distribution of Problem Ratings.

<img src="https://github.com/user-attachments/assets/24aa67c8-3e83-4ff9-b3cb-f7e248533c5f" width="400"/>

### Figures 3-6: Aggregated Confusion Matrices
These figures present the aggregated confusion matrices (row- and column-normalized) for each classifier:

#### Decision Tree
<img src="https://github.com/user-attachments/assets/ed0d7c45-7718-469c-b064-76aa51cf0da1" width="400"/>

#### Random Forest
<img src="https://github.com/user-attachments/assets/2fbe00cc-f098-4515-93e8-a2eb01700b9b" width="400"/>

#### KNN
<img src="https://github.com/user-attachments/assets/411e45dd-2981-4dc9-a83d-920bbd01c3b9" width="400"/>

#### SVM
<img src="https://github.com/user-attachments/assets/31dd1d0e-be70-451d-b2a5-bcd33d3f251f" width="400"/>

## Results
After running the script, you will see output similar to:

```
         Model          Accuracy
    _________________    ________
    {'Decision Tree'}     0.7891
    {'Random Forest'}    0.87775 
    {'KNN'          }    0.83945 
    {'SVM'          }    0.83537 
```

These results indicate the average accuracy across cross-validation folds for each model. You can further tune the hyperparameters or improve feature extraction to enhance model performance.

## Future Improvements

### Advanced Feature Extraction
Consider extracting features from the actual source code (if available), such as:
- Code length
- Loop nesting depth
- Cyclomatic complexity

### Hyperparameter Tuning
Use grid search or Bayesian optimization to fine-tune model parameters.

### Data Augmentation
Incorporate submissions from multiple users or contests to increase data diversity.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- **Dr. Osama Farouk**: For his invaluable insights and guidance in **Information Theory**.  
- **Dr. Haidy Saeed**: For her dedicated support and inspirational motivation throughout my learning journey.
