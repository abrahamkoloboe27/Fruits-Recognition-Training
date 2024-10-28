import requests
import os
import random
import json

API_URL = 'http://localhost:8000/'


# Choisir une image au hasard et l'envoyer à l'API pour prédiction 

def random_image_choice():
    os.chdir('data/Test')
    random_folder = random.choice(os.listdir())
    os.chdir(random_folder)
    random_image = random.choice(os.listdir())
    
    return random_folder, random_image

def send_image_to_api(folder, image):
    with open(image, 'rb') as f:
        files = {'file': f}
        response = requests.post(API_URL + 'predict', files=files)
        print(response.json())
        print(f"Image: {image} - Prediction: {response.json()['predicted_class']} - True label: {folder}")
        
        
if __name__ == '__main__':
    folder, image = random_image_choice()
    try : 
        send_image_to_api(folder, image)
    except:
        print("Error")