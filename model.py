# First install required packages
#This is to setup the FastChat model locally and generate responses using the model.


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def setup_fastchat_model(model_name="lmsys/fastchat-t5-3b-v1.0", cache_dir="./model_cache"):
    """
    Downloads and sets up the FastChat T5 model locally
    
    Args:
        model_name (str): Name of the model on Hugging Face
        cache_dir (str): Local directory to store the model
    
    Returns:
        tuple: (tokenizer, model) ready for inference
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Downloading model {model_name}...")
    
    # Download and load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=True
    )
    
    # Download and load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    print("Model downloaded and loaded successfully!")
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_length=128):
    """
    Generate a response using the loaded model
    
    Args:
        tokenizer: The loaded tokenizer
        model: The loaded model
        prompt (str): Input prompt
        max_length (int): Maximum length of the generated response
    
    Returns:
        str: Generated response
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    # Setup the model
    tokenizer, model = setup_fastchat_model()
    
    # Test the model
    test_prompt = "What is artificial intelligence?"
    print("\nTesting model with prompt:", test_prompt)
    response = generate_response(tokenizer, model, test_prompt)
    print("Response:", response)