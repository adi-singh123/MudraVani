# MudraVani - AI-Powered Indian Sign Language Recognition

MudraVani is an AI-driven system designed to **recognize and interpret Indian Sign Language (ISL)** using computer vision and machine learning. The project aims to bridge the communication gap between the Deaf and Hard of Hearing (DHH) community and the general population by providing **real-time hand gesture recognition** and **text-to-speech conversion** for seamless interaction.

---

🚀 Features

✔ Real-Time Sign Recognition – Uses AI to detect and interpret ISL signs instantly.

✔ Alphabet Recognition – Accurately identifies individual letters in ISL.

✔ Word Formation Capability – Converts sequences of gestures into meaningful words and sentences.

✔ Sign-to-Speech Conversion – Converts recognized gestures into audible speech output.

✔ Interactive UI – A clean, user-friendly React frontend for easy interaction.

✔ Lightweight & Fast – Optimized model for smooth performance on various devices.

---

🛠 Tech Stack

🔹 Frontend: React.js (for interactive UI)🔹 Backend: Python🔹 Machine Learning: MediaPipe, OpenCV, Scikit-learn, TensorFlow🔹 Database: MongoDB (Optional)

---


## 🎯 How It Works

1️⃣ **Hand Gesture Detection**: The system captures hand movements via a webcam.\
2️⃣ **Feature Extraction**: Keypoints from hand gestures are extracted using MediaPipe.\
3️⃣ **Model Prediction**: The trained ML model predicts the corresponding alphabet or sign.\
4️⃣ **Sentence Formation**: The system intelligently groups letters into words.\
5️⃣ **Speech Conversion (Optional)**: Converts recognized text into speech for better accessibility.

---

## 🖥️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
 git clone https://github.com/your-repo/mudravani.git
 cd mudravani
```

### 2️⃣ Install Dependencies

```bash
pip install -r backend/requirements.txt
cd frontend && npm install
```

### 3️⃣ Run the Backend (Python API)

```bash
cd backend
python app.py  # Or use uvicorn if FastAPI is used
```

### 4️⃣ Start the Frontend (React App)

```bash
cd frontend
npm start
```

### 5️⃣ Access MudraVani

Open [**http://localhost:3000/**](http://localhost:3000/) or your any given port number in your browser and start recognizing ISL signs! 🎉

---

## 🎯 Use Cases

✔ **Education & Learning**: Helps individuals learn Indian Sign Language.\
✔ **Assistive Technology**: Enables the DHH community to communicate easily.\
✔ **Customer Support**: Businesses can integrate ISL recognition for accessibility.\
✔ **Smart Devices**: Can be embedded in IoT devices for gesture-based controls.

---

## 📜 Roadmap

🔹 Improve accuracy for complex ISL words.\
🔹 Expand sign vocabulary beyond alphabets.\
🔹 Enhance UI for better user experience.\
🔹 Integrate sign-to-speech in multiple languages.

---

## 🤝 Contribution

We welcome contributions! If you'd like to improve MudraVani, please **fork the repository, create a feature branch, and submit a pull request.**

### Contributors 👥

- [Nihal Jaiswal] – AI/ML Expert
- [Aditya Singh](https://github.com/adi-singh123) – Frontend
- [Jiwanshu Kumar](https://github.com/jiwansh) – Backend
- [Prajjwal Srivatava] - UI/UX

---

## 📞 Contact

For queries, feel free to reach out at amanjiwanshu@gmail.com or open an issue in this repo.

🌟 **If you like this project, give it a star!** ⭐

