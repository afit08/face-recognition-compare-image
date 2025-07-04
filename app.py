from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import base64
import face_recognition
import numpy as np
import asyncio
import cv2

app = FastAPI()

SIMILARITY_THRESHOLD = 0.6

class FaceImages(BaseModel):
    image1: str
    image2: str
    image3: str
    image4: str

def fast_base64_to_rgb_np(base64_str: str) -> np.ndarray:
    try:
        image_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[base64 decode error] {e}")
        return None

def get_face_encodings(image_np: np.ndarray):
    try:
        if image_np is None:
            return []
        locations = face_recognition.face_locations(image_np, model='hog')
        if not locations:
            return []
        return face_recognition.face_encodings(image_np, locations)
    except Exception as e:
        print(f"[encoding error] {e}")
        return []

async def process_image(base64_img: str):
    return get_face_encodings(fast_base64_to_rgb_np(base64_img))

@app.post("/recognize")
async def recognize_attendance(images: FaceImages):
    # Parallel processing
    enc1, enc2, enc3, enc4 = await asyncio.gather(
        process_image(images.image1),
        process_image(images.image2),
        process_image(images.image3),
        process_image(images.image4),
    )

    if not enc1:
        return JSONResponse(status_code=200, content={
            "result": "Failure",
            "message": "No face found in image1.",
            "similarity_score": 0
        })

    if not (enc2 and enc3 and enc4):
        return JSONResponse(status_code=200, content={
            "result": "Failure",
            "message": "One or more reference images have no face.",
            "similarity_score": 0
        })

    comparison_encodings = enc2 + enc3 + enc4

    # Faster: early exit when match is found
    for query_enc in enc1:
        results = face_recognition.compare_faces(comparison_encodings, query_enc, tolerance=1 - SIMILARITY_THRESHOLD)
        if any(results):
            distances = face_recognition.face_distance(comparison_encodings, query_enc)
            score = 1 - min(distances)
            return JSONResponse(status_code=200, content={
                "result": "Success",
                "message": "Attendance marked.",
                "similarity_score": round(score, 4)
            })

    return JSONResponse(status_code=200, content={
        "result": "Failure",
        "message": "No matching faces found.",
        "similarity_score": 0
    })
