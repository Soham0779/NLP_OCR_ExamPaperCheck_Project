from flask import Flask, render_template, request
from PIL import Image
import pytesseract
import numpy as np
import gensim.downloader as api
import nltk
import io

app = Flask(__name__)

# Load the Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

# Download the 'punkt' resource for tokenization
nltk.download('punkt')

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity = None
    if request.method == 'POST':
        original_answer = request.form['original']
        
        # Process uploaded image
        student_answer = request.files['student_image']
        if student_answer:
            img_bytes = student_answer.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # Perform OCR using pytesseract
            student_text = pytesseract.image_to_string(img)
            
            # Tokenize text
            original_tokens = nltk.word_tokenize(original_answer.lower())
            student_tokens = nltk.word_tokenize(student_text.lower())

            # Remove out-of-vocabulary words
            original_tokens = [word for word in original_tokens if word in word2vec_model.key_to_index]
            student_tokens = [word for word in student_tokens if word in word2vec_model.key_to_index]

            # Calculate vectors for each text by averaging word vectors
            original_vector = np.mean([word2vec_model[word] for word in original_tokens], axis=0)
            student_vector = np.mean([word2vec_model[word] for word in student_tokens], axis=0)

            # Calculate cosine similarity
            similarity = np.dot(original_vector, student_vector) / (np.linalg.norm(original_vector) * np.linalg.norm(student_vector))
            similarity = similarity * 100  # Convert to percentage

    return render_template('index.html', similarity=similarity)

if __name__ == '__main__':
    app.run(debug=True)
