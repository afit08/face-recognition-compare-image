from flask import Flask, request, jsonify
import base64
import face_recognition
from io import BytesIO

app = Flask(__name__)

# Define a threshold for similarity score
SIMILARITY_THRESHOLD = 0.6

@app.route('/recognize', methods=['POST'])
def recognize_attendance():
    data = request.form
    
    # Decode base64 images
    image1_base64 = data['image1']
    image2_base64 = data['image2']
    image3_base64 = data['image3']
    image4_base64 = data['image4']
    
    image1 = base64.b64decode(image1_base64)
    image2 = base64.b64decode(image2_base64)
    image3 = base64.b64decode(image3_base64)
    image4 = base64.b64decode(image4_base64)
    
    # Load images
    image1_np = face_recognition.load_image_file(BytesIO(image1))
    image2_np = face_recognition.load_image_file(BytesIO(image2))
    image3_np = face_recognition.load_image_file(BytesIO(image3))
    image4_np = face_recognition.load_image_file(BytesIO(image4))
    
    # Find face locations and encodings
    face_locations1 = face_recognition.face_locations(image1_np)
    face_encodings1 = face_recognition.face_encodings(image1_np, face_locations1)
    face_locations2 = face_recognition.face_locations(image2_np)
    face_encodings2 = face_recognition.face_encodings(image2_np, face_locations2)
    face_locations3 = face_recognition.face_locations(image3_np)
    face_encodings3 = face_recognition.face_encodings(image3_np, face_locations3)
    face_locations4 = face_recognition.face_locations(image4_np)
    face_encodings4 = face_recognition.face_encodings(image4_np, face_locations4)
    
    # Compare faces
    for encoding1 in face_encodings1:
        for encoding2 in face_encodings2:
            for encoding3 in face_encodings3:
                for encoding4 in face_encodings4:
                    # Compare the face embeddings
                    face_distances = face_recognition.face_distance(encoding1, [encoding2, encoding3, encoding4])
                    similarity_score = 1 - min(face_distances)  # Convert distance to similarity score
                    
                    # Check if similarity score exceeds threshold
                    if similarity_score > SIMILARITY_THRESHOLD:
                        # Attendance marked
                        return jsonify({'result': 'Success', 'message': 'Attendance marked.', 'similarity_score': similarity_score})
    
    # No match found
    return jsonify({'result': 'Failure', 'message': 'No matching faces found.', 'similarity_score': 0})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8100)
