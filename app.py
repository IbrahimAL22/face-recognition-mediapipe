from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize drawing and face mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Custom drawing specifications for smaller points and segments
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Function to extract facial landmarks from an image
def get_landmarks(image, face_mesh):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            return [(lm.x, lm.y) for lm in face_landmarks.landmark]
    return None

# Function to compute the similarity between two sets of landmarks
def compute_similarity(landmarks1, landmarks2):
    if not landmarks1 or not landmarks2 or len(landmarks1) != len(landmarks2):
        return float('inf')

    # Normalize landmarks to a common reference frame
    landmarks1 = np.array(landmarks1)
    landmarks2 = np.array(landmarks2)

    # Compute the mean squared error
    mse = np.mean((landmarks1 - landmarks2) ** 2)
    return mse

# Function to process and draw landmarks on the frame
def process_frame(image, results, name):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw facial landmarks on the image with custom specifications
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
            # Get bounding box coordinates
            bbox = get_bounding_box(face_landmarks, image)
            # Draw the box with the name
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return image

# Function to get the bounding box coordinates around the face
def get_bounding_box(face_landmarks, image):
    if face_landmarks is None:
        return 0, 0, 0, 0  # Return default values if no landmarks detected

    # Initialize extreme points with large values
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    # Iterate through each landmark in NormalizedLandmarkList
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])

        # Update bounding box coordinates
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y

    return x_min, y_min, x_max, y_max

# Load reference image and extract landmarks
reference_image_path = 'image.jpg'  # Replace with your reference image path
reference_image = cv2.imread(reference_image_path)

# Initialize face mesh
with mp_face_mesh.FaceMesh() as face_mesh:
    reference_landmarks = get_landmarks(reference_image, face_mesh)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change 0 to video file path if needed

def generate_frames():
    # Create face mesh instance with default parameters (adjust as needed)
    with mp_face_mesh.FaceMesh() as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, check for end of stream
                continue

            # Flip the image for a more natural orientation
            image = cv2.flip(image, 1)

            # Extract landmarks from the current frame
            current_landmarks = get_landmarks(image, face_mesh)

            # Compute similarity between reference landmarks and current landmarks
            similarity = compute_similarity(reference_landmarks, current_landmarks)

            # Determine the name to display based on similarity
            if similarity < 0.01:  # Adjust threshold as needed
                name = "Ibrahim"
            else:
                name = "Unknown"

            # Process and draw landmarks with box and name
            annotated_image = process_frame(image.copy(), face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), name)

            # Encode the image to JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_image)
            frame = buffer.tobytes()

            # Yield the frame in the byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
