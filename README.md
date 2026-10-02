# Machine Learning final project

Predicting academic success in higher education. In this project I work with a dataset of 4,424 students and 37 variables (demographic, admission, socioeconomic, macroeconomic and per-semester performance) and set up three complementary tasks on the same data. The first is a multiclass classification of the student's final status (`abandono` = dropout, `matriculado` = enrolled, `graduado` = graduate). The second is a regression of the 2nd-semester average grade (`nota_media_2sem`), making sure no information from that same semester leaks in as a predictor. The third is a descriptive clustering to find academic-trajectory profiles.

Author: Pilar Pérez Hernández. Machine Learning, 2nd year IMAT B, ICAI – Universidad Pontificia Comillas, 2025/2026.

*[Versión en español](README.es.md)*


## Key results

Full numbers are in `memoria.pdf` (report, in Spanish); these are the ones that summarise each task.

- **Classification** (tuned Gradient Boosting, test set): macro F1 **0.693**, accuracy 0.767, weighted F1 0.758. The trivial baseline that always predicts "graduado" gets macro F1 0.22. "Matriculado" is the hard class (F1 0.44, AP 0.45 vs ~0.9 for the other two).
- **Regression** (tuned Random Forest, test set): RMSE **2.69** vs 5.20 for the `DummyRegressor` (48% reduction), MAE 1.46, R^2 0.73, using only information available at the end of the 1st semester.
- **Clustering** (KMeans on PCA at 90% variance, k=3): silhouette **0.39**, ARI vs Ward **0.89**, mean bootstrap ARI **0.91** (min 0.77). Three profiles: at-risk / mass dropout (n=856, 82% dropout), average performance (n=3289) and high performance with older age (n=279).

The same variables dominate all three tasks: subjects passed and average grade in the 1st semester, admission grade, tuition up to date, debtor status and age at enrolment.

<p align="center">
  <img src="figures/02_roc_pr_curves.png" alt="One-vs-rest ROC and Precision-Recall curves of the classifier" width="800">
</p>
<p align="center">
  <img src="figures/02_shap_summary.png" alt="SHAP summary plot per class (Random Forest)" width="520">
</p>
<p align="center">
  <img src="figures/03_residuals.png" alt="Predicted vs actual and residuals of the regression" width="800">
</p>
<p align="center">
  <img src="figures/04_tsne_clusters.png" alt="t-SNE of the PCA space coloured by cluster (k=3) and by real outcome" width="800">
</p>


## Folder contents

```
.
├── 01_EDA_final.ipynb              EDA and the decisions that shape everything else
├── 02_Clasificacion_final.ipynb    Task 1: multiclass classification
├── 03_Regresion_final.ipynb        Task 2: regression of nota_media_2sem
├── 04_No_supervisado_final.ipynb   Task 3: clustering + PCA + t-SNE
├── rendimiento_estudiantes.csv     Data (separator `;`)
├── memoria.pdf                     Report in NeurIPS format, 8 pages max (Spanish)
├── figures/                        Figures extracted from the notebooks for this README
├── README.md                       This file
└── README.es.md                    Spanish version
```

Notebooks, report and in-code comments are written in Spanish.


## How to reproduce the results

The notebooks are not chained together because there are no intermediate artefacts to pass from one to the next, but it is worth reading them in order. The reason is that many of the decisions I apply in the models (which variables I exclude, which metric I use, why k=3 in clustering) are argued first in the EDA and then taken for granted in the other notebooks.

The first one, `01_EDA_final.ipynb`, looks at the target distribution, correlations, association between categorical variables with Cramér's V and dropout rates per degree programme. This is where I justify every decision that comes later. The second, `02_Clasificacion_final.ipynb`, compares eight base models with 5-fold CV, tries imbalance strategies (`class_weight='balanced'` and SMOTENC), tunes Random Forest and Gradient Boosting with `GridSearchCV`, evaluates on test (accuracy, macro F1, weighted F1, one-vs-rest ROC and PR), draws learning curves and analyses feature importance with Gini, permutation importance and SHAP. The third, `03_Regresion_final.ipynb`, repeats the scheme on `nota_media_2sem` excluding every 2nd-semester variable as a predictor (leakage control), reports RMSE, MAE and R^2, breaks the metrics down by target regime (mass at 0 vs continuous cloud between 10 and 18) and finishes with residuals, learning curves, importances and SHAP. And the fourth, `04_No_supervisado_final.ipynb`, clusters on PCA at 90% variance, compares KMeans and Ward for `k=3`, measures stability with bootstrap (20 resamples at 80%), visualises with t-SNE and compares against the `k=2` variant.

Each notebook can be run end to end with "Restart & Run All" without depending on the others. All of them fix `SEED = 42`, so the results should reproduce faithfully.

To re-run the four notebooks from the terminal without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute 01_EDA_final.ipynb           --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute 02_Clasificacion_final.ipynb --ExecutePreprocessor.timeout=2400
jupyter nbconvert --to notebook --execute 03_Regresion_final.ipynb     --ExecutePreprocessor.timeout=1200
jupyter nbconvert --to notebook --execute 04_No_supervisado_final.ipynb --ExecutePreprocessor.timeout=600
```


## Dependencies

Tested with Python 3.10. The reference versions in my environment are:

```
pandas   >= 1.5
numpy    >= 1.23
matplotlib >= 3.6
seaborn  >= 0.12
scikit-learn >= 1.3
scipy    >= 1.10
imbalanced-learn >= 0.11   # for SMOTENC in notebook 02
shap     >= 0.44           # for the SHAP summary plots in 02 and 03
```

To install everything at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy imbalanced-learn shap
```

I wrapped the calls to `imblearn` and `shap` in a `try/except ImportError`, so if either library is missing the notebooks detect it and skip those cells without breaking anything.


## Data

The CSV (`rendimiento_estudiantes.csv`) has to be in the same folder as the notebooks. It is read with `pd.read_csv(..., sep=';')` because the separator is not a comma, and it has 4,424 rows and 37 columns, with no missing values or duplicates. The target `objetivo` splits into graduado (49.9%), abandono (32.1%) and matriculado (17.9%), a moderate imbalance but enough to keep in mind when choosing the metric.

The file already comes with the categorical variables translated into readable labels, so columns like `ocupacion_madre`, `nacionalidad` or `cualificacion_previa` are text rather than numeric codes. That saves a recoding step but also means keeping an eye on the number of levels of some categoricals (`curso` has 17 values and `ocupacion_madre` reaches 31), because one-hot encoding them inflates the dimension quickly.


## Approximate running times

On a modern laptop with a regular CPU, the times I see are the following. The EDA takes one to two minutes, mostly because of the pairplots and the correlation matrix. Classification is the most expensive, five to ten minutes, and almost all of it goes into the `GridSearchCV` for RF and GB and the permutation importance. Regression is three to five minutes. And the unsupervised notebook takes two to three minutes, with t-SNE being the slowest part.


## Reproducibility and leakage control

I tried to keep everything stochastic under the same seed, `SEED = 42`, which applies to the splits, the stochastic models, the bootstrap in notebook 04 and t-SNE. In classification I use a stratified `train_test_split` to keep the proportions of the three classes in train and test; in regression I do not stratify because the target is continuous, but I do fix the split seed.

To avoid leaks, every preprocessing step (`StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`) is wrapped in a `Pipeline` or `ColumnTransformer`, so it is fitted only on the train part of each CV fold. That way there is no contamination from the test set or between folds. The SMOTENC in notebook 02 lives inside an `imblearn.Pipeline`, which means the oversampling is applied only to the training fold inside the CV and never to the validation fold.

In regression I also explicitly exclude the six contemporaneous 2nd-semester variables (`nota_media_2sem` and the five `asignaturas_2sem_*`) and the `objetivo` variable before the split. The justification is in section 2 of notebook 03 and also in the report.


## Implementation notes

One decision I took early was to work with two preprocessing pipelines depending on the model. For trees I use `OrdinalEncoder` with no scaling, because trees do not care about scale and do not need one-hot. For the linear models, KNN, SVM and MLP I set up `OneHotEncoder` plus `StandardScaler`, because there I was worried about introducing false orderings in categoricals or letting one variable dominate just because of its scale. The same logic applies in regression.

The main classification metric is macro F1, not accuracy. The reason is that under imbalance accuracy rewards always predicting the majority class without penalising ignoring the minority. Macro F1 weights the three classes equally and forces you to also take care of "matriculado" (17.9%), which is the hardest one. I report accuracy and weighted F1 as secondary metrics to have the full picture.

In regression the central metric is RMSE, complemented with MAE and R^2. Since the target is bimodal (a peak at 0 plus a continuous cloud between 10 and 18), I also report the metrics broken down by regime: if you only look at the global RMSE you hide the fact that the model gets the mass of zeros very well but is quite a bit noisier in the continuous range.

Clustering works on PCA at 90% variance to reduce noise and speed things up. I compare KMeans with hierarchical Ward for the same `k` (that lets me confirm the clusters are in the data and not in the algorithm) and measure stability with 20 bootstraps at 80%.


## Determinism

With the seed fixed, sklearn gives deterministic results within the same library version. The t-SNE output may vary slightly between scikit-learn versions, but it does not change the interpretation of the clusters or move the ARIs from one decile to another. If after installing the dependencies you get values very different from mine, the most likely cause is a fairly different scikit-learn version.


## Use of artificial intelligence

I used generative AI tools (ChatGPT and Claude) as support while preparing the project. Specifically, so it is clear how far that support goes: moving the report into the NeurIPS LaTeX template (structure of the `.tex`, tables with `booktabs`, `geometry` and `titlesec` tweaks); formatting the README in Markdown (section structure, code blocks); occasional technical questions about libraries such as specific scikit-learn parameters, `ColumnTransformer` syntax, integrating `imblearn.Pipeline` with `SMOTENC` or the right way to call `shap.TreeExplainer` for multiclass classifiers; and spotting typos and rephrasing passages that sounded clumsy to me.

The exploratory analysis, the choice of models, the methodological decisions (which variables to exclude for leakage, why macro F1 over accuracy, why k=3 in clustering), the interpretation of the results and the conclusions are my own work. The AI tools did not generate the notebooks or the analytical content of the report; they helped with editing and technical clarifications, the same way consulting documentation or a classmate would.
