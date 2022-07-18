import base64
import json
import os
import requests
from datetime import datetime

hostname = 'http://localhost:3000'
dir_path = os.path.join(os.path.dirname(__file__), 'image_captioning', 'images')


def image_captioning_test():
    file_path = os.path.join(dir_path, f'test_image_2.png')
    with open(file_path, 'rb') as img_file:
        base64_img = base64.b64encode(img_file.read()).decode('utf-8')

        img_data = {
            'image': f'data:image/jpeg;base64,{base64_img}',
        }

    url = f'{hostname}/image-captioning'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = json.dumps(img_data)

    response = requests.post(url, headers=headers, data=payload)

    print(response.text)


if __name__ == '__main__':
    # Test image captioning with "test_image_1.png"
    start_time = datetime.now()
    image_captioning_test()
    print(datetime.now() - start_time)
