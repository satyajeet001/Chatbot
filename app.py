# import cv2
# from scipy.signal import find_peaks, medfilt, detrend
# from flask import Flask, render_template, send_from_directory, request, jsonify
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from scipy.signal import find_peaks, medfilt

# app = Flask(__name__, static_url_path='/static')

# nltk.download("punkt")

# with open("Database_questions.txt", "r") as file:
#     conversation = file.read()

# sentences = nltk.sent_tokenize(conversation)
# word_tokens = [nltk.word_tokenize(sent) for sent in sentences]

# tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
# tfidf_matrix = tfidf_vectorizer.fit_transform(word_tokens)

# # tfidf_vectorizer = TfidfVectorizer(lowercase=False)
# # tfidf_matrix = tfidf_vectorizer.fit_transform(word_tokens)


# def generate_response(user_input):
#     user_input = user_input.lower()
#     word_tokens_input = nltk.word_tokenize(user_input)
#     tfidf_vector_input = tfidf_vectorizer.transform([word_tokens_input])
#     similarities = cosine_similarity(tfidf_matrix, tfidf_vector_input)
#     most_similar_index = similarities.argmax()
#     return sentences[most_similar_index]


# def preprocess_ecg(ecg_signal):
#     # Median filtering to remove high-frequency noise
#     filtered_ecg = medfilt(ecg_signal, kernel_size=9)

#     # Smoothing to maintain P and T waves
#     smoothed_ecg = np.convolve(filtered_ecg, np.ones(15)/15, mode='same')

#     return smoothed_ecg


# def detect_r_peaks(ecg_signal):
#     peaks, _ = find_peaks(ecg_signal, height=0.5,
#                           distance=50)  # Adjust parameters
#     return peaks


# def check_for_atrial_fibrillation(time_diffs_seconds):
#     for i in range(len(time_diffs_seconds) - 1):
#         time_diff = time_diffs_seconds[i]
#         next_time_diff = time_diffs_seconds[i + 1]
#         diff_between_diffs = abs(next_time_diff - time_diff)

#         if diff_between_diffs > 0.5:
#             return True

#     return False


# def create_mask(image):
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
#     mask = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
#     )
#     return mask


# @app.route('/')
# def home():
#     return render_template('home.html')


# @app.route('/chat')
# def chat():
#     return render_template('index.html')


# @app.route("/get_response", methods=["POST"])
# def get_response():
#     user_input = request.form["user_input"]
#     response = generate_response(user_input)
#     return jsonify({"response": response})


# @app.route('/upload_image', methods=['POST'])
# def upload_image():
#     if 'image' not in request.files:
#         return jsonify({"message": "No image part"}), 400

#     image = request.files['image']

#     # Save the uploaded image temporarily
#     uploaded_image_path = 'uploaded_image.png'
#     image.save(uploaded_image_path)

#     # Read and process the uploaded image
#     ecg_image = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)

#     # Your image processing and ECG analysis code here
#     cropped_ecg = ecg_image[10:500, 50:1000]  # Adjust the cropping region
#     resized_ecg = cv2.resize(cropped_ecg, (500, 300))  # Adjust the dimensions
#     mask = create_mask(resized_ecg)
#     cleaned_ecg = cv2.bitwise_and(resized_ecg, resized_ecg, mask=mask)

#     # Sum over vertical axis to get a 1D signal
#     ecg_signal = cleaned_ecg.sum(axis=0)
#     r_peak_indices = detect_r_peaks(ecg_signal)

#     preprocessed_ecg = preprocess_ecg(ecg_signal)  # Preprocess the ECG signal

#     # Calculate time differences between consecutive R peaks
#     time_diffs = np.diff(r_peak_indices)
#     time_diffs_seconds = time_diffs / 1000.0

#     atrial_fibrillation = check_for_atrial_fibrillation(time_diffs_seconds)

#     if atrial_fibrillation:
#         return render_template('unhealthy.html')
#     else:
#         return render_template('healthy.html')

#     # return jsonify({"message": analysis_result}), 200


# # if __name__ == '__main__':p
# #     app.run(debug=True)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80)






import cv2
import numpy as np
from scipy.signal import find_peaks, medfilt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def create_mask(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    return thresholded

def preprocess_ecg(ecg_signal):
    filtered_ecg = medfilt(ecg_signal, kernel_size=9)
    smoothed_ecg = np.convolve(filtered_ecg, np.ones(15)/15, mode='same')
    return smoothed_ecg

def detect_r_peaks(ecg_signal):
    peaks, _ = find_peaks(ecg_signal, height=20, distance=50)
    return peaks

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"message": "No image part"}), 400

    image = request.files['image']

    uploaded_image_path = 'uploaded_image.png'
    image.save(uploaded_image_path)

    ecg_image = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)
    cropped_ecg = ecg_image[10:500, 50:1000]
    resized_ecg = cv2.resize(cropped_ecg, (500, 300))
    mask = create_mask(resized_ecg)
    cleaned_ecg = cv2.bitwise_and(resized_ecg, resized_ecg, mask=mask)
    enhanced_ecg = cv2.convertScaleAbs(cleaned_ecg, alpha=1.5, beta=0)
    ecg_signal = enhanced_ecg.sum(axis=0)

    r_peak_indices = detect_r_peaks(ecg_signal)
    preprocessed_ecg = preprocess_ecg(ecg_signal)

    time_per_mm = 40
    amplitude_per_box = 0.1

    time_values = np.arange(len(preprocessed_ecg)) * time_per_mm
    amplitude_values = preprocessed_ecg * amplitude_per_box
    time_values -= time_values.min()
    amplitude_values -= amplitude_values.min()
    amplitude_values = -amplitude_values
    amplitude_values += 3500

    local_maxima_indices, _ = find_peaks(amplitude_values, distance=50, prominence=150)
    time_diffs = np.diff(time_values[local_maxima_indices])
    time_diffs_seconds = time_diffs / 1000.0

    atrial_fibrillation = False
    for i in range(len(time_diffs_seconds) - 1):
        time_diff = time_diffs_seconds[i]
        next_time_diff = time_diffs_seconds[i + 1]
        diff_between_diffs = abs(next_time_diff - time_diff)
        if diff_between_diffs > 0.5:
            atrial_fibrillation = True
            break

    if atrial_fibrillation:
        result_message = "You have atrial fibrillation"
    else:
        result_message = "You are healthy"

    return render_template('result.html', message=result_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
