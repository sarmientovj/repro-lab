# Reproducible Script for a Language Model

## What this is
Training script for a language model using sklearn.model_selection utilities, 
though note that deep learning LMs typically use PyTorch/TensorFlow. 
This example shows how to integrate sklearn's cross-validation and hyperparameter 
search with a custom LM training loop.

## Data
Dynamic data generation for reproducibility
The dataset used in this programm is generated dynamically. The sample data consists of text in Quechua, an indigenous language widely spoken in the Peruvian Andes.

## Reproduce the result
1. pip install -r requirements.txt
2. python src/train.py --seed 42

## Expected output
        ✅ All seeds set to: 42

        🚀 Training Configuration
        Model: bert-base-uncased
        Device: cpu
        Batch Size: 16
        Epochs: 3
        Learning Rate: 2e-05
        Max Length: 128
        Seed: 42
        ============================================================

        📊 Creating sample dataset...
        Dataset size: 500
        Class distribution: [168 167 165]

        ✂️ Splitting data with seed...
        Train size: 350
        Validation size: 75
        Test size: 75

        🔧 Loading model and tokenizer...
        Loading weights: 100% 199/199 [00:00<00:00, 415.89it/s, Materializing param=bert.pooler.dense.weight]BertForSequenceClassification LOAD REPORT from: bert-base-uncased
        Key                                        | Status     | 
        -------------------------------------------+------------+-
        cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
        cls.seq_relationship.weight                | UNEXPECTED | 
        cls.predictions.transform.dense.bias       | UNEXPECTED | 
        cls.predictions.bias                       | UNEXPECTED | 
        cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
        cls.seq_relationship.bias                  | UNEXPECTED | 
        cls.predictions.transform.dense.weight     | UNEXPECTED | 
        classifier.weight                          | MISSING    | 
        classifier.bias                            | MISSING    | 

        Notes:
        - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
        - MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

        🎯 Starting training...
        ============================================================
        Epoch 1/3 [Train]: 100%|██████████| 22/22 [07:00<00:00, 19.10s/it, loss=1.0962]
        Evaluating: 100%|██████████| 5/5 [00:25<00:00,  5.19s/it]

        Epoch 1/3
        Train Loss: 1.1149 | Train Acc: 0.3086
        Val Loss: 1.1026 | Val Acc: 0.3333
        --------------------------------------------------
        Epoch 2/3 [Train]: 100%|██████████| 22/22 [06:49<00:00, 18.63s/it, loss=1.1185]
        Evaluating: 100%|██████████| 5/5 [00:25<00:00,  5.20s/it]

        Epoch 2/3
        Train Loss: 1.1057 | Train Acc: 0.3400
        Val Loss: 1.1012 | Val Acc: 0.3067
        --------------------------------------------------
        Epoch 3/3 [Train]: 100%|██████████| 22/22 [06:48<00:00, 18.59s/it, loss=1.1407]
        Evaluating: 100%|██████████| 5/5 [00:25<00:00,  5.16s/it]

        Epoch 3/3
        Train Loss: 1.0925 | Train Acc: 0.3714
        Val Loss: 1.1017 | Val Acc: 0.3067
        --------------------------------------------------

        📈 Final Evaluation on Test Set
        ============================================================
        Evaluating: 100%|██████████| 5/5 [00:25<00:00,  5.17s/it]

        ✅ Training Complete!
        Test Loss: 1.1033
        Test Accuracy: 0.2667

        📋 Classification Report:
                    precision    recall  f1-score   support

            Negative       0.29      0.08      0.12        25
            Positive       0.23      0.28      0.25        25
            Neutral       0.30      0.44      0.35        25

            accuracy                           0.27        75
        macro avg       0.27      0.27      0.24        75
        weighted avg       0.27      0.27      0.24        75


        ============================================================
        Testing Reproducibility
        ============================================================

        Run 1:
        ✅ All seeds set to: 42
        Test Accuracy: 0.400000

        Run 2:
        ✅ All seeds set to: 42
        Test Accuracy: 0.400000

        ✅ Reproducible! Both runs got 0.400000

        ============================================================
        Training script completed successfully!
        ============================================================

## Environment
Python 3.11; exact packages in requirements.txt; see Dockerfile.
