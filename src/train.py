import numpy as np
import torch
import random
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. Set Seed for Reproducibility
# ============================================
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✅ All seeds set to: {seed}")

# ============================================
# 2. Custom Dataset Class
# ============================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================
# 3. Training Function
# ============================================
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs=3):
    """Basic training loop"""
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        train_predictions = []
        train_labels = []

        # Training phase
        model.train()
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in train_bar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Get predictions
            preds = torch.argmax(outputs.logits, dim=1)
            train_predictions.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Update progress bar
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate training metrics
        train_acc = accuracy_score(train_labels, train_predictions)
        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("-" * 50)

    return model

def evaluate_model(model, data_loader, device):
    """Evaluation function"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)

    return avg_loss, accuracy

# ============================================
# 4. Prepare Sample Data
# ============================================
def create_sample_data(seed=42, n_samples=1000):
    """Create sample text classification data"""
    texts = [ # This is a sample text for classification purpose number
        f"Kayqa huk muestra qillqam clasificación propósito yupaypaq {i}"
        for i in range(n_samples)
    ]
    # Add some variety
    for i in range(n_samples):
        if i % 3 == 0:   # I love this product! It's amazing and works great.
            texts[i] = "¡Anchatan munakuni kay ruruta! Admirakuypaqmi hinaspapas ancha allintam llamkan." + texts[i]
        elif i % 3 == 1: # Terrible experience, would not recommend to anyone.
            texts[i] = "Manchay experiencia, mana pimanpas recomendasaqchu." + texts[i]
        else:            # It's okay, nothing special but gets the job done.
            texts[i] = "Allinmi, manan imapas especialchu aswanpas llank’ayta hunt’achin." + texts[i]

    # Create synthetic labels (0: negative, 1: positive, 2: neutral)
    # labels = np.random.randint(0, 3, n_samples)
    # Create labels (0: negative, 1: positive, 2: neutral)
    labels = np.random.RandomState(seed).randint(0, 3, n_samples)

    return texts, labels

# ============================================
# 5. Main Training Script
# ============================================
def main( SEED=42 ):
    # Set seed first thing
    set_seed(SEED)

    # Configuration
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n🚀 Training Configuration")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Max Length: {MAX_LENGTH}")
    print(f"  Seed: {SEED}")
    print("="*60)

    # Create sample data
    print("\n📊 Creating sample dataset...")
    n_samples = 500

    texts, labels = create_sample_data(SEED, n_samples)

    print(f"  Dataset size: {n_samples}")
    print(f"  Class distribution: {np.bincount(labels)}")

    # Split data with fixed seed
    print("\n✂️ Splitting data with seed...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=SEED, stratify=labels )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp )

    print(f"  Train size: {len(X_train)}")
    print(f"  Validation size: {len(X_val)}")
    print(f"  Test size: {len(X_test)}")

    # Initialize tokenizer and model
    print(f"\n🔧 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3 ).to(DEVICE)

    # Create datasets
    train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle is fine with seed set
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Train the model
    print("\n🎯 Starting training...")
    print("="*60)

    trained_model = train_model(
        model, train_loader, val_loader, optimizer, scheduler, DEVICE, EPOCHS )

    # Final evaluation on test set
    print("\n📈 Final Evaluation on Test Set")
    print("="*60)
    test_loss, test_acc = evaluate_model(trained_model, test_loader, DEVICE)

    print(f"\n✅ Training Complete!")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    # Detailed classification report
    print("\n📋 Classification Report:")
    trained_model.eval()
    all_preds = []
    all_labels_test = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = trained_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())

    print(classification_report(
        all_labels_test, all_preds,
        target_names=['Negative', 'Positive', 'Neutral']
    ))

    return trained_model

# ============================================
# 6. Verify Reproducibility
# ============================================
def test_reproducibility( SEED=42 ):
    """Run training twice to verify same results"""
    print("\n" + "="*60)
    print("Testing Reproducibility")
    print("="*60)

    results = []
    for run in range(2):
        print(f"\nRun {run+1}:")

        # Set seed
        set_seed(SEED)

        # Create identical data
        texts = ["Sample text " + str(i) for i in range(100)]
        labels = np.random.RandomState(SEED).randint(0, 2, 100)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=SEED
        )

        # Simple model for testing
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(random_state=SEED)
        model.fit(X_train_vec, y_train)

        accuracy = model.score(X_test_vec, y_test)
        results.append(accuracy)

        print(f"  Test Accuracy: {accuracy:.6f}")

    if results[0] == results[1]:
        print(f"\n✅ Reproducible! Both runs got {results[0]:.6f}")
    else:
        print(f"\n⚠️ Not reproducible: {results[0]} vs {results[1]}")

    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    # # Run main training
    main( ap.parse_args().seed )

    # # Optional: Test reproducibility
    test_reproducibility( ap.parse_args().seed )

    # for seed in [13, 21, 42, 87, 100]:
    #  main(seed)
    #  test_reproducibility(seed)
    #  print("*"*80)

    print("\n" + "="*60)
    print("Training script completed successfully!")
    print("="*60)
