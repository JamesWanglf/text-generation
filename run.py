import base64
import io
import torch
from datetime import datetime
from flask import Flask, request, Response, jsonify, make_response
from PIL import Image
from image_captioning.models.blip import blip_decoder
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

app = Flask(__name__)
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_PROCESS_OK = 100
IMAGE_PROCESS_ERR = 101
INVALID_REQUEST_ERR = 231
INVALID_IMAGE_ERR = 232
UNKNOWN_ERR = 500

ERR_MESSAGES = {
    IMAGE_PROCESS_OK: 'The image is processed successfully.',
    IMAGE_PROCESS_ERR: 'The image process has been failed.',
    INVALID_REQUEST_ERR: 'Invalid request.',
    INVALID_IMAGE_ERR: 'Invalid image has input. Could not read the image data.',
    UNKNOWN_ERR: 'Unknown error has occurred.'
}


def transform_raw_image(raw_image, image_size=384):
    encoded_data = raw_image.split(',')[1]
    decoded_string = io.BytesIO(base64.b64decode(encoded_data))
    img = Image.open(decoded_string).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(img).unsqueeze(0).to(device)
    return image


def load_model():
    global model

    image_size = 384
    model = blip_decoder(pretrained='image_captioning/checkpoints/model_base.pth', image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)


def process_image(raw_image):
    global model

    # Transform the raw_image
    image = transform_raw_image(raw_image)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        print('caption: ' + caption[0])

    return IMAGE_PROCESS_OK, caption[0]


@app.route('/', methods=['GET'])
def welcome():
    """
    Welcome page
    """
    return Response("<h1 style='color:red'>Image Captioning server is running!</h1>", status=200)


@app.route('/image-captioning', methods=['GET', 'POST'])
def face_recognition():
    if request.method == 'GET':
        return Response('Image Captioning  server is running.', status=200)

    # POST
    # Read image data
    img_data = request.json
    if 'image' not in img_data:
        response = {
            'error': ERR_MESSAGES[INVALID_REQUEST_ERR]
        }
        return make_response(jsonify(response), 400)

    # Load model
    if model is None:
        load_model()

    # Process image
    start_time = datetime.now()
    res_code, caption = process_image(img_data['image'])
    print(f'Image process takes {datetime.now() - start_time}')

    if res_code != IMAGE_PROCESS_OK:
        response = {
            'error': ERR_MESSAGES[res_code]
        }
        return make_response(jsonify(response), 500)

    # Return candidates
    response = {
        'caption': caption
    }
    return make_response(jsonify(response), 200)
