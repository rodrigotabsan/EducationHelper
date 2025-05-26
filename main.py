import streamlit as st
from app.config import get_openai_api_key
from app.document_processor import DocumentProcessor
from app.rag_engine import RAGEngine
from app.session_manager import init_session
from app.ui_components import render_css

def main():
    st.set_page_config(page_title="📚 Tutor Inteligente RAG", layout="wide")
    init_session()
    render_css()

    st.title("📚 Tutor Inteligente RAG")
    st.markdown("### Tu asistente personal para el aprendizaje basado en documentos")

    # Sidebar
    with st.sidebar:
        st.subheader("⚙️ Configuración")
        api_key = st.text_input("🔑 OpenAI API Key", value=get_openai_api_key(), type="password")
        knowledge_level = st.selectbox("🎓 Nivel de Conocimiento", ["Principiante", "Intermedio", "Avanzado"])
    
    # Documento
    uploaded_file = st.file_uploader("📁 Sube tu documento", type=["pdf", "docx", "pptx"])

    if uploaded_file and api_key and not st.session_state.document_processed:
        with st.spinner("Procesando documento..."):
            try:
                processor = DocumentProcessor()
                text = processor.process(uploaded_file)

                engine = RAGEngine()
                chunks = engine.split_text(text)
                vectorstore = engine.create_vectorstore(chunks)
                prompt = engine.get_prompt()
                qa_chain = engine.build_qa_chain(vectorstore, prompt)

                st.session_state.vectorstore = vectorstore
                st.session_state.qa_chain = qa_chain
                st.session_state.document_processed = True
                st.success("✅ Documento procesado correctamente")
            except Exception as e:
                st.error(str(e))

    # Preguntas
    if st.session_state.document_processed:
        question = st.text_area("🤔 ¿Qué te gustaría aprender?")
        if st.button("🚀 Preguntar") and question:
            enhanced_question = f"Nivel de conocimiento: {knowledge_level}. Pregunta: {question}"
            response = st.session_state.qa_chain.run(enhanced_question)
            st.session_state.chat_history.append((question, response))
            st.rerun()

        for i, (q, r) in enumerate(st.session_state.chat_history):
            st.markdown(f"**🧑‍🎓 Pregunta {i+1}:** {q}")
            st.markdown(f"**👨‍🏫 Respuesta:** {r}")
            st.markdown("---")

if __name__ == "__main__":
    main()
