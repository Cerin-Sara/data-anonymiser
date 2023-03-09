import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the PII categories to identify
pii_categories = ['NAME', 'ADDRESS', 'EMAIL', 'PHONE', 'SSN']

# Example text to classify
text = "John Doe's address is 123 Main Street, New York, NY 10001 and his email is john.doe@example.com"

# Tokenize the text and add special tokens for BERT
input_ids = tokenizer.encode(text, add_special_tokens=True)

# Convert the input to a PyTorch tensor
input_tensor = torch.tensor([input_ids])

# Use the model to make a prediction
outputs = model(input_tensor)

# Get the predicted PII category
predicted_category_idx = torch.argmax(outputs[0]).item()
predicted_category = pii_categories[predicted_category_idx]

# Print the predicted category
print("Predicted PII category:", predicted_category)
