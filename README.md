📚 Multi-PDF ChatApp AI Agent 🤖
💬 Chat Seamlessly with Multiple PDFs

A smart AI-powered application that lets you upload multiple PDFs and chat with them instantly.
Built with LangChain, Google Gemini Pro, and FAISS Vector Database, and deployed using Streamlit for a clean and interactive UI.

🧠 Features

✅ Upload and process multiple PDFs at once
✅ Extract and embed text using FAISS Vector DB
✅ Query PDFs using Google Gemini Pro
✅ Instant, context-aware responses
✅ Beautiful and responsive Streamlit UI

🧰 Tech Stack

Python 3.10+

LangChain (for text embedding & retrieval)

Google Gemini Pro (LLM model via API)

FAISS (for vector search)

Streamlit (for web app deployment)

PyPDF2 / pdfminer.six (for PDF text extraction)

dotenv (for API key handling)

⚙️ Setup & Installation
1️⃣ Clone or Download the Repository
git clone https://github.com/yourusername/multi-pdf-chatapp.git
cd multi-pdf-chatapp

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Your Google Gemini API Key

Create a .env file in the project root and add this line:

GOOGLE_API_KEY=your_api_key_here


(You can get your key from Google AI Studio
)

▶️ Run the Application
streamlit run app.py


Then open your browser and go to:
👉 http://localhost:8501/

🧩 How It Works

Upload multiple PDF files.

App extracts all text and embeds it using FAISS Vector DB.

When you ask a question, LangChain retrieves the relevant context.

Google Gemini Pro LLM generates accurate answers based on your PDFs.

📁 Project Structure
multi_pdf_chatapp/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
├── .env                  # Google API key
├── venv/                 # Virtual environment
└── data/                 # (Optional) PDFs folder

🚀 Future Enhancements

Add memory for multi-turn conversations

Support for DOCX & TXT files

Export chat history

Deploy to Hugging Face or Streamlit Cloud

🧑‍💻 Author

Khushi Chauhan
