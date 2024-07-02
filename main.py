import numpy as np
import cv2
import torch
import albumentations as albu

import base64
import image as image_utils
import model as model_utils

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = model_utils.create_model()
model.eval();

def get_mask_warped_img(dataUri):
    image, width, height = image_utils.load_image(dataUri)
    image = cv2.resize(image, (854, 480))

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = image_utils.pad(image, factor=32, border=cv2.BORDER_CONSTANT)

    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(image_utils.tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = image_utils.unpad(mask, pads)

    imgMask = (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (255, 255, 255)).astype(np.uint8)
    imgMask = cv2.resize(imgMask, (854, 480))

    warped, points = image_utils.extract_idcard(padded_image, imgMask)

    extracted_area = cv2.countNonZero(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))

    assert extracted_area > (0.07 * (854*480))

    return(imgMask, warped, points)


@app.route('/process', methods=['POST'])
def process_data_uri():
    try:
        data_uri = request.json.get('data_uri')

        if not data_uri:
            return jsonify({"error": "No data_uri provided"}), 400

        try:
            imgMask, warped, points = get_mask_warped_img(data_uri)
        except Exception as e:
            return jsonify({"error": str(e), "message": "ID card not found in the image"}), 500
        
        new_points = [[int(i[0]), int(i[1])] for i in points]
        
        IDmaskDataUri = "data:image/png;base64," + str(base64.b64encode(cv2.imencode('.png', imgMask)[1]).decode('utf-8'))
        extractedIDDataUri = "data:image/png;base64," + str(base64.b64encode(cv2.imencode('.png', warped)[1]).decode('utf-8'))

        return jsonify({"IDmaskDataUri":IDmaskDataUri, "extractedIDDataUri":extractedIDDataUri, "points":new_points}), 200

    except Exception as e:
        return jsonify({"error": str(e), "message": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(port=3000)
