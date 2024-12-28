import sys
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Path model T5 yang sudah di-finetune
model_path = './chatbot_final_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Pilih device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def answer_question(question: str) -> str:
    """
    Fungsi untuk menghasilkan jawaban menggunakan model T5
    yang sudah di-finetune.
    """
    # Prompt yang lebih eksplisit
    prompt = (
        "You are an AI assistant. Please provide a helpful, detailed answer "
        "to the following question:\n\n"
        + question
        + "\nAnswer:\n"
    )

    # Tokenisasi prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)

    # Generate jawaban
    outputs = model.generate(
        inputs,
        max_length=128,
        min_length=20,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    # Dekode menjadi string
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

@app.route('/')
def index():
    """
    Endpoint untuk menampilkan halaman index.html
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint untuk menerima pertanyaan (dikirim dari front-end)
    dan mengembalikan jawaban (JSON).
    """
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'answer': 'Pertanyaan kosong.'}), 400

    # Cetak ke console untuk debugging
    print(f"[LOG] Pertanyaan pengguna: {user_message}", flush=True)

    # Proses pertanyaan dengan model
    answer = answer_question(user_message)

    # Kembalikan jawaban dalam format JSON
    return jsonify({'answer': answer})

if __name__ == '__main__':
    print("Menjalankan aplikasi Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)
