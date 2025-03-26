import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import MNISTModel

class MNISTInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTModel().to(self.device)
        self.model.load_state_dict(torch.load('models/mnist_model.pth', map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image).unsqueeze(0)

    def predict(self, image):
        with torch.no_grad():
            tensor = self.preprocess_image(image).to(self.device)
            output = self.model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = output.argmax(1).item()
            confidence = probabilities[0][prediction].item()
        return prediction, confidence

if __name__ == "__main__":
    # Initialize the inference class
    mnist_classifier = MNISTInference()
    
    # Load and process an image
    try:
        # You can replace 'test_image.png' with your image path
        image_path = 'test_image.png'
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Make prediction
        prediction, confidence = mnist_classifier.predict(image)
        
        print(f"Predicted digit: {prediction}")
        print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
