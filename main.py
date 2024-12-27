#This is the main file that will be used to run the chatbot


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class FastChatT5:
    def __init__(self, model_path="./model_cache/lmsys/fastchat-t5-3b-v1.0"):
        """
        Initialize the FastChat T5 model using a local path
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    
    def generate_response(self, prompt, max_length=128, temperature=0.7):
        """
        Generate a response for a given prompt
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def batch_generate(self, prompts, max_length=128, temperature=0.7):
        """
        Generate responses for multiple prompts at once
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
        
        responses = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        return responses

# Example 1: Basic usage
def example_basic_usage():
    # Initialize the model
    chatbot = FastChatT5()
    
    # Single prompt
    prompt = "Explain quantum computing in simple terms"
    response = chatbot.generate_response(prompt)
    print(f"Prompt: {prompt}\nResponse: {response}\n")

# Example 2: Batch processing
def example_batch_processing():
    chatbot = FastChatT5()
    
    # Multiple prompts
    prompts = [
        "What is machine learning?",
        "How does a neural network work?",
        "Explain deep learning"
    ]
    
    responses = chatbot.batch_generate(prompts)
    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}\nResponse: {response}\n")

# Example 3: Integration in a web application
from flask import Flask, request, jsonify

app = Flask(__name__)
chatbot = FastChatT5()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    response = chatbot.generate_response(prompt)
    return jsonify({'response': response})

# Example 4: Using in a custom pipeline
def custom_conversation_pipeline(context, question):
    chatbot = FastChatT5()
    
    # Combine context and question into a prompt
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = chatbot.generate_response(prompt, max_length=200)
    return response

if __name__ == "__main__":
    # Run basic example
    print("Running basic example:")
    example_basic_usage()
    
    # Run batch processing example
    print("\nRunning batch processing example:")
    example_batch_processing()
    
    # Example of custom pipeline usage
    print("\nRunning custom pipeline example:")
    context = "The Earth orbits around the Sun in an elliptical path."
    question = "Why doesn't the Earth fall into the Sun?"
    answer = custom_conversation_pipeline(context, question)
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Uncomment to run the Flask web server
    # app.run(debug=True)