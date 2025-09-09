import csv
import os
import base64
from openai import AzureOpenAI
from diseases import diseases

from tqdm import tqdm

IM_FOLDER = 'image_jpeg'
IMAGES = os.listdir(IM_FOLDER)
CSV_PATH = 'majority_voted.csv'

MODEL = 'gpt-4.1'

with open(CSV_PATH, 'r') as f:
    r = csv.DictReader(f)
    rows = list(r)
rows = [row for row in rows if f"{row['StudyInstanceUID']}.JPEG" in IMAGES]

endpoint = os.getenv("ENDPOINT_URL", "https://shohomc.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", MODEL)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-01-01-preview",
)

# IMAGE_PATH = "YOUR_IMAGE_PATH"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

def get_prompt(b64_img):
    with open("prompt.txt", "r") as f:
        return [
            {
                "role": "user",
                "content": [
                    {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}"
                    }  
                    },
                    {
                        "type": "text",
                        "text": f.read()
                    }
                ]
                }
        ]


messages = []

for path in tqdm(IMAGES, desc="Sending requests: "): 
    uid = path.replace(".JPEG", "").strip()
    path = os.path.join(IM_FOLDER, path)
    with open(path, 'rb') as f: 
        row = [row for row in rows if row["StudyInstanceUID"] == uid][0]
        
        b64_image = base64.b64encode(f.read()).decode("utf-8")
        prompt = get_prompt(b64_image)
        
        letters = []
        for letter, disease in diseases.items(): 
            if disease in row['answer']: 
                letters.append(letter)
        
        response = client.chat.completions.create(
            model=deployment,
            messages=prompt,
            max_tokens=800,
            temperature=0.0,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        
        response = response.choices[0].message.content 
        
        choices = response.split(",")
        
        data = [
            {'UID': uid, 'true': str(letters), 'response': str(choices)}
        ]
         
        with open(f'multichoice/{MODEL}.csv', 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['UID', 'true', 'response'])
            writer.writerows(data)
