# Quantized Model

This folder contains all resources, code, and documentation related to the quantization and knowledge distillation of BERT-based models for phishing URL detection.

## Contents

- **Dataset_(phishing_site_urls)_prep.ipynb**  
  Prepares and balances the phishing URL dataset. Downloads data from [Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls/data), cleans and balances it, and splits it into train, validation, and test sets. The processed dataset is then uploaded to the Hugging Face Hub as `KaushiGihan/phishing-site-url`.

- **Fine tune_LLM_Bert_Teacher_model.ipynb**  
  Trains a BERT-based "teacher" model for binary classification (Safe/Not Safe) on the phishing URL dataset.  
  - Uses Hugging Face Transformers and Datasets.
  - Freezes most of the base BERT layers except the pooler and classifier.
  - Evaluates using accuracy and ROC AUC.
  - Pushes the trained model to the Hugging Face Hub as `KaushiGihan/bert-text-classifier`.

- **Fine tune_LLM_Bert_student_model.ipynb**  
  Implements knowledge distillation to train a smaller "student" model (DistilBERT) using the teacher model's outputs as soft targets.  
  - Drops layers and attention heads for efficiency.
  - Uses a combination of distillation loss (KL divergence) and hard label loss (cross-entropy).
  - Evaluates both teacher and student models on test and validation sets.
  - Pushes the distilled model to the Hugging Face Hub as `KaushiGihan/bert-text-classifier_KD_model`.

## Workflow Summary

1. **Data Preparation**  
   - Download and clean the phishing URL dataset.
   - Balance the dataset (equal samples of "good" and "bad" URLs).
   - Split into train/validation/test and upload to Hugging Face Hub.

2. **Teacher Model Training**  
   - Fine-tune a BERT model for phishing URL classification.
   - Evaluate and push the model to the Hub.

3. **Student Model Distillation & Quantization**  
   - Train a smaller DistilBERT model using knowledge distillation.
   - Evaluate and push the quantized/distilled model to the Hub.

## Results

- The teacher model achieves high accuracy and ROC AUC on the validation set.
- The student (distilled) model achieves comparable or better performance with reduced size and faster inference.

## Usage

- Use the provided notebooks to reproduce the data preparation, training, and distillation process.
- Download the final models from the Hugging Face Hub for deployment or further research. 