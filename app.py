from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import base64
import face_recognition
from io import BytesIO
import numpy as np

app = FastAPI()

# Define a threshold for similarity score
SIMILARITY_THRESHOLD = 0.6

class FaceImages(BaseModel):
    image1: str
    image2: str
    image3: str
    image4: str

def decode_and_get_encodings(base64_str: str):
    try:
        image_data = base64.b64decode(base64_str)
        image_np = face_recognition.load_image_file(BytesIO(image_data))
        face_locations = face_recognition.face_locations(image_np)
        return face_recognition.face_encodings(image_np, face_locations)
    except Exception:
        return []

@app.post("/recognize")
async def recognize_attendance(images: FaceImages):
    encodings1 = decode_and_get_encodings(images.image1)
    encodings2 = decode_and_get_encodings(images.image2)
    encodings3 = decode_and_get_encodings(images.image3)
    encodings4 = decode_and_get_encodings(images.image4)

    if not encodings1:
        return JSONResponse(status_code=200, content={"result": "Failure", "message": "No face found in image1.", "similarity_score": 0})

    for encoding1 in encodings1:
        for encoding2 in encodings2:
            for encoding3 in encodings3:
                for encoding4 in encodings4:
                    face_distances = face_recognition.face_distance(encoding1, [encoding2, encoding3, encoding4])
                    similarity_score = 1 - min(face_distances) if face_distances.size > 0 else 0

                    if similarity_score > SIMILARITY_THRESHOLD:
                        return JSONResponse(status_code=200, content={
                            "result": "Success",
                            "message": "Attendance marked.",
                            "similarity_score": similarity_score
                        })

    return JSONResponse(status_code=200, content={
        "result": "Failure",
        "message": "No matching faces found.",
        "similarity_score": 0
    })
