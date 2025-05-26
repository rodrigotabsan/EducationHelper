# EducationHelper
An interactive educational application based on RAG (Retrieval-Augmented Generation) that uses artificial intelligence to generate explanations adapted to the user's knowledge level, from PDF, DOCX, or PPTX documents.

## 🚀 Main Features

* Document upload: PDF, Word (.docx) and PowerPoint (.pptx)
* Automatic content extraction
* Knowledge level selector: Beginner, Intermediate, Advanced
* Personalized explanation generation with LangChain + OpenAI
* Modern and responsive interface with Streamlit
* Conversation history per session
* Clear pedagogical structure: introduction, development with examples and final summary


## 📦 Requirements
1. Clone the repository

git clone https://github.com/rodrigotabsan/educationhelper.git
cd intelligent-tutor-rag

2. Create a virtual enviroment

- Windows
python -m venv venv
venv\Scripts\activate

- macOS / Linux
python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. How to use the app

streamlit run app.py

5. Scheme of the educative prompt

The AI will respond under the following framework:

* Professional Scope: Education
* Action: Explain with examples and create study notes
* Steps:
 1) Adaptation to user's level
 2) Selection of the best outline or summary
 3) Recommendation to search for additional sources
* Context: Guided learning
* Restrictions: Always adapt the response to the shown context
* Template: Introduction + Explanation with examples + Final summary
