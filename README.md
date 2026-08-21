#  Երեխեք թասկերը նայեք  Issues -ի մեջ,  սրան ուշադրություն մի դարձրեք  զուտ թողել եմ կարողա համեմատենք հետո , վերջում կփոխենք ReadMe -ն



# Cassava Leaf Disease Classification

## Project Overview

This project aims to classify cassava leaf images into five classes:

- CBB — Cassava Bacterial Blight
- CBSD — Cassava Brown Streak Disease
- CGM — Cassava Green Mottle
- CMD — Cassava Mosaic Disease
- Healthy

## Project Goal

Build and evaluate deep learning models for automatic cassava leaf disease classification using image data.

## Team

- Suzanna Makaryan — Data & Preprocessing
- Victoria Margaryan — Baseline & Transfer Learning
- Ruzanna Torosyan — Model Improvement
- Christine Mkhitaryan — Evaluation & Explainability


## Team Project Roadmap

### Project Goal

Build an image classification model that can identify the condition of a cassava leaf from an image.

The model must classify each image into one of 5 classes:

1. CBB — Cassava Bacterial Blight
2. CBSD — Cassava Brown Streak Disease
3. CGM — Cassava Green Mottle
4. CMD — Cassava Mosaic Disease
5. Healthy

---

# TEAM STRUCTURE

### Person 1 — Data & Preprocessing

Responsible for:

* Dataset understanding
* Data exploration
* Data quality
* Class distribution
* Train/validation split
* Image preprocessing
* Data augmentation

### Person 2 — Baseline & Transfer Learning

Responsible for:

* Baseline CNN
* Baseline training
* Transfer learning
* First pretrained models
* Training experiments

### Person 3 — Model Improvement

Responsible for:

* Hyperparameter tuning
* Class imbalance
* Fine-tuning
* Model comparison
* Improving the best model

### Person 4 — Evaluation & Explainability

Responsible for:

* Model evaluation
* Confusion matrix
* Error analysis
* Misclassified images
* Grad-CAM / explainability
* Final model analysis

IMPORTANT:
The project must remain ONE pipeline.
Nobody should build an isolated project that cannot be connected to the others.

---

# PHASE 1 — DATASET UNDERSTANDING & EDA

## Owner: Person 1

### Goal

Understand exactly what data we have before training any model.

---

## Step 1.1 — Download and inspect the dataset

Tasks:

* Download the Kaggle dataset.
* Identify all files and folders.
* Identify where the training images are stored.
* Identify the training metadata file.
* Open the training CSV and inspect its columns.

Questions to answer:

* How many images are available?
* What is the name of the image identifier column?
* What is the name of the target/label column?
* How are images connected to their labels?

Deliverable:

* Short description of the dataset structure.

---

## Step 1.2 — Identify the target classes

Find all unique labels.

Confirm that there are 5 classes:

* CBB
* CBSD
* CGM
* CMD
* Healthy

Create a clear mapping between numerical labels and disease names.

Deliverable:

* Class mapping table.

Example:

| Label | Disease |
| ----- | ------- |
| 0     | CBB     |
| 1     | CBSD    |
| 2     | CGM     |
| 3     | CMD     |
| 4     | Healthy |

---

## Step 1.3 — Analyze class distribution

Count how many images belong to each class.

Create:

* class frequency table
* bar chart

Question:

> Is the dataset balanced?

If it is not balanced, document which classes are underrepresented and which class is dominant.

This observation will be used later when deciding how to handle class imbalance.

Deliverable:

* Class distribution table
* Class distribution visualization
* Short conclusion

---

## Step 1.4 — Visualize representative images

For every class:

* randomly select several images
* display them
* show their class name

Goal:

Understand visually what the classes look like.

Questions:

* Can humans visually distinguish the diseases?
* Are there visible symptoms?
* Are some diseases visually similar?
* Are healthy leaves clearly different?
* Are backgrounds different between images?

Deliverable:

A grid of representative images for all 5 classes.

---

## Step 1.5 — Analyze image properties

Inspect:

* image width
* image height
* number of channels
* image format
* minimum/maximum dimensions

Questions:

* Are all images the same size?
* Do image dimensions vary?
* Are all images RGB?
* Are there unusual images?

Deliverable:

A short image-property report.

---

## Step 1.6 — Check data quality

Investigate:

* missing labels
* missing image files
* corrupted images
* duplicated records
* unexpected file formats
* invalid labels

Do NOT automatically delete anything.

First identify the problem and document it.

Deliverable:

Data-quality report.

---

## Step 1.7 — Final EDA conclusions

Write a short conclusion answering:

1. What is the dataset?
2. How many images do we have?
3. How many classes?
4. Is the dataset balanced?
5. Are the images consistent in size?
6. Are there data-quality problems?
7. What problems might affect model training?

### PHASE 1 OUTPUT

Person 1 must provide:

* EDA notebook
* Dataset description
* Class distribution
* Representative image visualization
* Image properties analysis
* Data-quality analysis
* EDA conclusions

---

# PHASE 2 — DATA PREPROCESSING

## Owner: Person 1

### Goal

Transform raw images into a form that can be used by neural networks.

---

## Step 2.1 — Define train/validation strategy

Split the labeled dataset into:

* training set
* validation set

Use a stratified strategy so that the class proportions remain similar in both sets.

Important:

The validation set must NOT be used for training.

Deliverable:

* Final train/validation split
* Number of images in each split
* Class distribution in both splits

---

## Step 2.2 — Define image size

Choose the input size required by the selected model.

For example:

`224 × 224`

The exact size should be determined by the architecture we use.

Questions:

* Why did we choose this size?
* What happens if images are resized?

Deliverable:

* Defined input image shape.

---

## Step 2.3 — Image normalization

Define how pixel values will be normalized.

The normalization must be compatible with the pretrained model if transfer learning is used.

Deliverable:

* Documented normalization strategy.

---

## Step 2.4 — Data augmentation

Design realistic augmentations.

Possible transformations:

* horizontal flip
* rotation
* crop
* zoom
* brightness change
* contrast change

Important:

Augmentation should represent realistic variations that could occur when photographing cassava leaves.

Do not create unrealistic images.

---

## Step 2.5 — Visualize augmentation

Take several original images and show how they look after augmentation.

Goal:

Verify that augmentation does not destroy important disease characteristics.

Deliverable:

Original vs augmented image visualization.

---

## Step 2.6 — Final preprocessing pipeline

Define the complete pipeline:

Raw image

↓

Resize

↓

Normalization

↓

Augmentation during training

↓

Model input

Validation images should use preprocessing but should NOT receive random training augmentation.

### PHASE 2 OUTPUT

* Final preprocessing pipeline
* Train/validation split
* Augmentation strategy
* Before/after augmentation visualization
* Short explanation of preprocessing decisions

---

# PHASE 3 — BASELINE CNN

## Owner: Person 2

### Goal

Build a simple first model to establish a baseline.

The purpose is NOT to achieve the highest score.

The purpose is to answer:

> Can a basic CNN learn the cassava disease classification task?

---

## Step 3.1 — Define baseline architecture

Create a relatively simple CNN.

Conceptually:

Image

↓

Convolution layers

↓

Pooling

↓

Feature extraction

↓

Fully connected layer

↓

5-class output

---

## Step 3.2 — Train the baseline

Train the CNN using:

* training set
* validation set

Track:

* training loss
* validation loss
* training accuracy
* validation accuracy

---

## Step 3.3 — Analyze learning curves

Create:

* training loss vs epochs
* validation loss vs epochs
* training accuracy vs epochs
* validation accuracy vs epochs

Questions:

* Is the model learning?
* Is it overfitting?
* Is it underfitting?
* At what point does validation performance stop improving?

---

## Step 3.4 — Evaluate baseline

Calculate at minimum:

* accuracy
* precision
* recall
* F1-score

Do not only report accuracy.

---

## Step 3.5 — Save baseline results

Create a table:

| Model        | Accuracy | Precision | Recall |  F1 |
| ------------ | -------: | --------: | -----: | --: |
| Baseline CNN |      ... |       ... |    ... | ... |

This table will later be expanded with all other models.

### PHASE 3 OUTPUT

* Baseline CNN
* Training curves
* Evaluation metrics
* Baseline result table
* Short analysis of model behavior

---

# PHASE 4 — TRANSFER LEARNING

## Owner: Person 2

### Goal

Use a pretrained image classification model instead of training everything from scratch.

---

## Step 4.1 — Choose pretrained architectures

Start with 2 architectures.

Recommended candidates:

* ResNet50
* EfficientNet

Do not test 10 different architectures immediately.

---

## Step 4.2 — Understand pretrained weights

Understand:

* What does pretrained mean?
* On what type of dataset was the model originally trained?
* What knowledge does the model already contain?
* Why can pretrained features help with our problem?

This explanation must be included in the presentation.

---

## Step 4.3 — Replace the classification head

The pretrained model originally predicts its own classes.

We need to replace the final classification layer so that it predicts:

5 cassava classes.

Conceptually:

Pretrained model

↓

Feature extractor

↓

New classifier

↓

CBB / CBSD / CGM / CMD / Healthy

---

## Step 4.4 — First transfer-learning experiment

Initially:

* freeze most pretrained layers
* train the new classification head

Record:

* training accuracy
* validation accuracy
* validation loss
* F1-score

---

## Step 4.5 — Compare with baseline

Add the result to the model table.

| Model        | Accuracy |  F1 |
| ------------ | -------: | --: |
| Baseline CNN |      ... | ... |
| ResNet50     |      ... | ... |
| EfficientNet |      ... | ... |

Question:

> Did transfer learning improve performance?

---

## Step 4.6 — Analyze training behavior

Compare:

* training curves
* validation curves
* convergence speed
* final validation performance

### PHASE 4 OUTPUT

* At least 2 pretrained models
* Training results
* Model comparison
* Explanation of why transfer learning helped or did not help

---

# PHASE 5 — MODEL IMPROVEMENT

## Owner: Person 3

### Goal

Take the best model so far and systematically improve it.

IMPORTANT:

Do not randomly change parameters.

Every experiment must answer a specific question.

---

## Step 5.1 — Investigate class imbalance

Based on Phase 1, determine whether minority classes are being ignored.

Test possible strategies:

* class weights
* weighted loss
* oversampling
* targeted augmentation

Do not apply all techniques at once.

Test them separately when possible.

---

## Step 5.2 — Compare augmentation strategies

Compare:

Experiment A:

* minimal augmentation

Experiment B:

* standard augmentation

Experiment C:

* stronger augmentation

Measure the effect on validation performance.

---

## Step 5.3 — Fine-tuning

Take the best pretrained model.

Initially:

* frozen backbone

Then:

* unfreeze selected deeper layers
* train with a smaller learning rate

Compare:

Frozen model

vs

Fine-tuned model

---

## Step 5.4 — Hyperparameter experiments

Test important parameters such as:

* learning rate
* batch size
* number of epochs
* optimizer
* weight decay
* dropout

Do not test everything randomly.

Create an experiment table.

Example:

| Experiment | Change            | Result |
| ---------- | ----------------- | -----: |
| Exp 1      | Baseline          |    ... |
| Exp 2      | More augmentation |    ... |
| Exp 3      | Class weights     |    ... |
| Exp 4      | Fine-tuning       |    ... |

---

## Step 5.5 — Select best model

Choose the model based on validation performance.

Do not choose based only on accuracy.

Consider:

* F1-score
* macro F1
* per-class recall
* confusion matrix

### PHASE 5 OUTPUT

* Experiment table
* Best hyperparameters
* Best model
* Explanation of why it was selected

---

# PHASE 6 — MODEL COMPARISON

## Owner: Person 3

### Goal

Perform a clean comparison of all important experiments.

---

## Step 6.1 — Create final comparison table

Include:

* Baseline CNN
* ResNet
* EfficientNet
* fine-tuned model
* best augmentation experiment
* best class imbalance strategy

Example:

| Model        | Accuracy | Macro F1 | Best/Worst Class | Notes             |
| ------------ | -------: | -------: | ---------------- | ----------------- |
| CNN          |      ... |      ... | ...              | Baseline          |
| ResNet50     |      ... |      ... | ...              | Transfer learning |
| EfficientNet |      ... |      ... | ...              | Transfer learning |
| Best model   |      ... |      ... | ...              | Fine-tuned        |

---

## Step 6.2 — Compare computational cost

If possible record:

* training time
* number of parameters
* model size
* inference time

This helps answer:

> Is the best model also practical?

---

## Step 6.3 — Select final candidate

Choose one model for final evaluation.

Document:

* why this model was selected
* what alternatives were tested
* why they were rejected

### PHASE 6 OUTPUT

A final model comparison and one selected candidate model.

---

# PHASE 7 — EVALUATION & ERROR ANALYSIS

## Owner: Person 4

### Goal

Understand exactly where the final model succeeds and fails.

---

## Step 7.1 — Final evaluation

Evaluate the selected model on the validation/test data.

Calculate:

* accuracy
* precision
* recall
* F1-score
* macro F1

---

## Step 7.2 — Confusion matrix

Create a 5 × 5 confusion matrix.

Analyze:

* Which class is easiest?
* Which class is hardest?
* Which two diseases are most often confused?
* Is Healthy being confused with diseases?
* Are minority classes suffering more errors?

---

## Step 7.3 — Per-class analysis

For each class calculate:

* precision
* recall
* F1

Then write a short interpretation.

Example:

> The model performs well on CMD but struggles to distinguish CBSD from CGM.

Do NOT just present numbers.

Explain what the numbers mean.

---

## Step 7.4 — Find misclassified images

Collect examples where:

* prediction ≠ actual label

For each image show:

* original image
* actual class
* predicted class
* confidence/probability if available

---

## Step 7.5 — Analyze errors

Look for patterns.

Possible reasons:

* similar symptoms
* poor image quality
* leaf partially visible
* background interference
* lighting
* disease symptoms are subtle
* multiple symptoms appear together

This section is very important for the final presentation.

---

# PHASE 8 — EXPLAINABILITY + FINAL MODEL + KAGGLE

## Owner: Person 4 + ALL TEAM

---

## Step 8.1 — Grad-CAM / Explainability

Use an explainability method such as Grad-CAM.

Goal:

Understand which part of the leaf influenced the model prediction.

For several examples show:

Original image

↓

Model prediction

↓

Grad-CAM heatmap

Questions:

* Is the model looking at the leaf?
* Is it focusing on disease symptoms?
* Is it focusing on the background?
* Does the model appear to learn meaningful visual features?

---

## Step 8.2 — Final model retraining

After selecting the best architecture and settings:

* define the final configuration
* train the final model using the agreed training strategy
* save the model
* document all final parameters

The final model should be reproducible.

---

## Step 8.3 — Generate predictions

Use the final model to generate predictions for the competition test images.

Verify:

* correct image IDs
* correct class labels
* correct output format
* correct number of predictions

---

## Step 8.4 — Kaggle submission

Create the required submission file.

Before submitting:

* check file structure
* check number of rows
* check labels
* check image IDs
* make sure there are no missing predictions

Submit to Kaggle.

Record:

* submission score
* model version
* experiment/configuration used

---

## Step 8.5 — Final result

Create one final table:

| Metric              | Result |
| ------------------- | -----: |
| Validation Accuracy |    ... |
| Validation Macro F1 |    ... |
| Kaggle Score        |    ... |
| Final Model         |    ... |

---

# 📊 FINAL PROJECT STRUCTURE

The GitHub repository should eventually look approximately like this:

```text
cassava-leaf-disease-classification/
│
├── README.md
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_transfer_learning.ipynb
│   ├── 05_model_improvement.ipynb
│   ├── 06_model_comparison.ipynb
│   └── 07_evaluation_explainability.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── models/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── submissions/
│
└── requirements.txt
```

---

# 🔗 HOW THE 4 PEOPLE CONNECT THEIR WORK

## Person 1 → Person 2

Person 1 gives:

* cleaned dataset
* train/validation split
* preprocessing strategy
* augmentation strategy

↓

Person 2 uses exactly these to build the baseline and transfer-learning models.

---

## Person 2 → Person 3

Person 2 gives:

* baseline results
* pretrained model results
* training curves
* candidate best models

↓

Person 3 improves these models.

---

## Person 3 → Person 4

Person 3 gives:

* best model
* final configuration
* model predictions
* validation results

↓

Person 4 performs evaluation and error analysis.

---

## Person 4 → ALL TEAM

Person 4 gives:

* confusion matrix
* classification report
* error examples
* Grad-CAM
* final model analysis

↓

Everyone uses these results for the final presentation.

---

# 🏆 FINAL PRESENTATION STORY

The presentation should tell one continuous story:

### 1. Problem

Cassava is an important food crop, but diseases can significantly affect production.

↓

### 2. Dataset

We have images of cassava leaves belonging to 5 classes.

↓

### 3. Data Analysis

We analyzed the dataset and found class imbalance and other characteristics.

↓

### 4. Preprocessing

We prepared and augmented the images.

↓

### 5. Baseline

We built a simple CNN as our starting point.

↓

### 6. Transfer Learning

We tested pretrained architectures.

↓

### 7. Improvement

We experimented with augmentation, class imbalance handling and fine-tuning.

↓

### 8. Evaluation

We analyzed accuracy, F1, confusion matrix and individual classes.

↓

### 9. Explainability

We investigated where the model focuses using Grad-CAM.

↓

### 10. Final Model

We selected the best-performing model.

↓

### 11. Kaggle

We generated predictions and submitted them to the competition.

↓

### 12. Conclusion

We explain:

* what worked
* what did not work
* where the model makes mistakes
* what could be improved in the future

---

# ✅ TEAM RULES

* [ ] Do not start modeling before understanding the dataset.
* [ ] Do not use the validation set for training.
* [ ] Do not compare models using accuracy only.
* [ ] Record every experiment.
* [ ] Every experiment must have a reason.
* [ ] Keep the same train/validation split when comparing models.
* [ ] Save important results.
* [ ] Document decisions.
* [ ] Do not delete problematic data without documenting why.
* [ ] Do not make four independent notebooks that cannot be connected.
* [ ] Every person must understand the whole pipeline, not only their own part.

---

# 🎯 FINAL DELIVERABLES

At the end, the team should have:

* [ ] Clean and documented dataset pipeline
* [ ] EDA
* [ ] Preprocessing pipeline
* [ ] Baseline CNN
* [ ] Transfer-learning models
* [ ] Model improvement experiments
* [ ] Model comparison
* [ ] Final model
* [ ] Confusion matrix
* [ ] Classification report
* [ ] Error analysis
* [ ] Grad-CAM/explainability
* [ ] Kaggle submission
* [ ] GitHub repository
* [ ] Final presentation
* [ ] Clear explanation of every major technical decision
