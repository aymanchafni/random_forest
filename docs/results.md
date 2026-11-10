# Results

Train 734 / test 184 patients (stratified, seed 42).

| Model | CV ROC AUC | CV accuracy | Test accuracy | Test recall | Test precision | Test ROC AUC |
|---|---|---|---|---|---|---|
| Majority class (baseline) | 0.500 ± 0.000 | 0.553 ± 0.003 | 0.554 | 1.000 | 0.554 | 0.500 |
| Logistic regression | 0.928 ± 0.039 | 0.857 ± 0.039 | 0.891 | 0.922 | 0.887 | 0.932 |
| Random forest, depth 2 (original) | 0.919 ± 0.034 | 0.849 ± 0.035 | 0.848 | 0.892 | 0.843 | 0.915 |
| Random forest, tuned | 0.932 ± 0.029 | 0.861 ± 0.035 | 0.891 | 0.941 | 0.873 | 0.936 |

Tuned forest hyper-parameters: `{'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 5}`

Most important features (permutation, test set): ST_Slope, ChestPainType, Cholesterol, Sex, ExerciseAngina
