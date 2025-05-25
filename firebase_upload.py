import os
import json
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase Admin
cred = credentials.Certificate('firebase_key.json')  # path to your Firebase service account JSON
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://projectkitty-9979b-default-rtdb.firebaseio.com/'  # replace with your DB URL
})

def upload_json_to_firebase(folder_path, category):
    ref = db.reference(f'pose_data/{category}')
    for file in os.listdir(folder_path):
        if file.endswith('_processed.json'):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            key = os.path.splitext(file)[0]  # filename without extension
            ref.child(key).set(data)
            print(f'Uploaded {file} to Firebase under pose_data/{category}/{key}')

if __name__ == "__main__":
    upload_json_to_firebase('scripts/cuts/goodb', 'goodb')
    upload_json_to_firebase('scripts/cuts/badb', 'badb')
    print("Upload complete!")
