# Import necessary libraries
import pandas as pd  # For handling Excel files and data
import torch  # For PyTorch, which powers BERT
from transformers import BertTokenizer, BertForSequenceClassification  # For BERT model and tokenizer
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import accuracy_score  # For evaluating accuracy
import numpy as np  # For numerical operations
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

# Check if GPU is available
print("Checking GPU availability...")
if torch.cuda.is_available():
    print(f"GPU is available! Device: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("GPU is not available. Using CPU instead.")

# Step 1: Load the data
file_path = 'benchmark-test.xlsx'  # Example: 'benchmark_data.xlsx'
sheet_name = 'Sheet1'  # Change this if your benchmark is on a different sheet

df = pd.read_excel(file_path, sheet_name=sheet_name)  # Load the Excel file

# Step 2: Extract relevant columns and handle missing data
df = df[['Headline', 'Text', 'sentiment']]  # Select columns

# Clean the data
df['combined_text'] = df['Headline'].fillna('') + ' ' + df['Text'].fillna('')  # Concatenate, filling NaN with empty string
df = df.dropna(subset=['combined_text', 'sentiment'])  # Drop rows where combined_text or Sentiment is missing
df = df[df['combined_text'] != ' ']  # Drop rows where combined_text is just empty space

# Debug: Print the shape of the dataframe after cleaning
print(f"After cleaning, dataframe has {len(df)} rows.")

# Map Sentiment labels to numbers, making it case-insensitive by converting to lowercase
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment_lower'] = df['sentiment'].str.lower()  # Convert to lowercase
df['label'] = df['sentiment_lower'].map(sentiment_map)  # Map the lowercase version
df = df.dropna(subset=['label'])  # Drop rows with unmapped sentiments

# Debug: Print the final shape and some sample data
print(f"After mapping sentiments, dataframe has {len(df)} rows.")
print(df.head())  # Print the first few rows to inspect

# If the dataframe is empty at this point, stop and check your data
if len(df) == 0:
    print("Error: No valid data left. Please check your Excel file for matching 'sentiment' values and ensure there are rows with data.")
else:
    # Step 3: Split the data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"Training set: {len(train_df)} rows, Testing set: {len(test_df)} rows")


# Step 4: Prepare the data for BERT
# Load the BERT tokenizer (Polish specific)
tokenizer = BertTokenizer.from_pretrained('dkleczek/bert-base-polish-cased-v1')


# Function to tokenize the text
def tokenize_data(dataframe, max_length=512):  # max_length is BERT's limit; we'll truncate longer texts
    inputs = tokenizer(
        list(dataframe['combined_text']),  # List of texts to tokenize
        padding=True,  # Pad shorter texts to the same length
        truncation=True,  # Truncate longer texts
        max_length=max_length,
        return_tensors='pt'  # Return as PyTorch tensors
    )
    labels = torch.tensor(dataframe['label'].values)  # Convert labels to a tensor
    return inputs, labels


# Tokenize training and testing data
train_inputs, train_labels = tokenize_data(train_df)
test_inputs, test_labels = tokenize_data(test_df)

# Step 5: Set up the BERT model
# We're using BertForSequenceClassification for sentiment analysis (it has 3 classes: negative, neutral, positive)
model = BertForSequenceClassification.from_pretrained(
    'dkleczek/bert-base-polish-cased-v1',
    num_labels=3  # 3 classes: negative (0), neutral (1), positive (2)
)

# Move model to GPU if available (faster training)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Lower LR, add weight decay
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # New: LR scheduler

# Step 6: Fine-tune the model (this is where training happens)
# For now, we'll just outline the training loop. You can run this part after your meeting if needed.
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW  # Optimizer for BERT

# Create DataLoaders for training and testing
train_dataset = TensorDataset(train_inputs['input_ids'].to(device),
                              train_inputs['attention_mask'].to(device),
                              train_labels.to(device))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Batch size=16; changed to 8

test_dataset = TensorDataset(test_inputs['input_ids'].to(device),
                             test_inputs['attention_mask'].to(device),
                             test_labels.to(device))
test_loader = DataLoader(test_dataset, batch_size=32)

# Training loop
optimizer = AdamW(model.parameters(), lr=5e-5)  # Learning rate; common for BERT
model.train()  # Set model to training mode

# Early stopping setup
best_accuracy = 0.0  # Track the best test accuracy
patience = 3  # Number of epochs to wait for improvement (adjust as needed, e.g., 2-5)
no_improve_count = 0  # Counter for epochs without improvement

num_epochs = 10  # Increase epochs here (e.g., from 5 to 10)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for batch in train_loader:  # Training batches
        optimizer.zero_grad()
        inputs, masks, labels = batch
        outputs = model(inputs, attention_mask=masks, labels=labels)  # Include labels for loss
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    scheduler.step()  # Update LR

    # Evaluation after each epoch (moved here for clarity)
    model.eval()  # Set model to evaluation mode
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:  # Use test_loader for validation
            inputs, masks, labels = batch
            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Epoch {epoch+1}/{num_epochs}: Test Accuracy: {accuracy * 100:.2f}%')
    
    # Early stopping logic
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        no_improve_count = 0  # Reset counter if improved
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best accuracy: {best_accuracy * 100:.2f}%")
            break  # Stop training

# After training, you can print final results or save the model
print(f"Training complete. Best accuracy achieved: {best_accuracy * 100:.2f}%")


