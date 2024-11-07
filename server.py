import os
import torch
from litserve import LitAPI, LitServer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CustomLitAPI(LitAPI):

    def setup(self, device):
        """
        Load the tokenizer and model
        """
        model_name = os.getenv("MODEL_NAME", "annyalvarez/Roberta-MoralPres-MS-DM-P2" )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
        self.model.to(device)
        self.model.eval()
	
        self.label_map = {
            0: "non-moral",
            1: "care/harm",
            2: "fairness/cheating",
            3: "loyalty/betrayal",
            4: "authority/subversion",
            5: "purity/degradation"}

    
    def decode_request(self, request):
        """
        Preprocess the request data (tokenize)
        """
        inputs = self.tokenizer(request["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        return inputs


    def predict(self, inputs):
        """
        Perform the inference
        """
        with torch.no_grad():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        return outputs.logits
    

    def encode_response(self, logits):
        """
        Process the model output into a response dictionary
        """
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        response = {}
        for i in range(len(probabilities[0])):
            label_name = self.label_map[i]
            response[label_name] = probabilities[0][i].item()  # Convert to Python float
        
        return response


if __name__ == "__main__":
    api = CustomLitAPI()
    server = LitServer(api, accelerator='cuda' if torch.cuda.is_available() else 'cpu', devices=1)
    server.run(port=8000)


