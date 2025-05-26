import streamlit as st
import os
from app.config import get_openai_api_key
from app.document_processor import DocumentProcessor
from app.rag_engine import RAGEngine
from app.session_manager import init_session
from app.ui_components import render_css

def main():
    # Configuración de página DEBE ir primero
    st.set_page_config(
        page_title="📚 Tutor Inteligente RAG",
        page_icon="📚", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session()
    render_css()
    
    # Título con estilo CSS
    st.markdown('<h1 class="main-header">📚 Tutor Inteligente RAG</h1>', unsafe_allow_html=True)
    st.markdown("### Tu asistente personal para el aprendizaje basado en documentos")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="knowledge-level">', unsafe_allow_html=True)
        st.markdown("## ⚙️ Configuración")
        
        # API Key
        api_key = st.text_input("🔑 OpenAI API Key", value=get_openai_api_key(), type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.markdown("---")
        
        # Nivel de conocimiento
        st.markdown("### 🎓 Nivel de Conocimiento")
        knowledge_level = st.selectbox(
            "Selecciona tu nivel:",
            ["Principiante", "Intermedio", "Avanzado"],
            help="Este nivel determinará la complejidad de las explicaciones"
        )
        
        # Descripción del nivel
        level_descriptions = {
            "Principiante": "📚 Explicaciones básicas con ejemplos simples",
            "Intermedio": "🔍 Explicaciones detalladas con ejemplos prácticos", 
            "Avanzado": "🎯 Explicaciones profundas con análisis crítico"
        }
        st.info(level_descriptions[knowledge_level])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Layout principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Carga de documento
        st.markdown('<div class="file-upload">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📁 Cargar Documento</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Sube tu documento:",
            type=["pdf", "docx", "pptx"],
            help="Formatos: PDF, Word, PowerPoint"
        )
        
        if uploaded_file and api_key and not st.session_state.document_processed:
            with st.spinner("🔄 Procesando documento..."):
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
                    st.info(f"📊 Archivo: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif not api_key:
            st.warning("⚠️ Ingresa tu API Key de OpenAI")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Estado del sistema
        if st.session_state.document_processed:
            st.success("🟢 Sistema listo")
        else:
            st.info("🔵 Esperando configuración")
    
    with col2:
        # Chat interface
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">💬 Conversación Educativa</h3>', unsafe_allow_html=True)
        
        # Mostrar historial
        if st.session_state.chat_history:
            for i, (q, r) in enumerate(st.session_state.chat_history):
                st.markdown(f"**🧑‍🎓 Pregunta {i+1}:** {q}")
                st.markdown(f"**👨‍🏫 Respuesta:**\n{r}")
                st.markdown("---")
        
        # Input para preguntas
        if st.session_state.document_processed and api_key:
            question = st.text_area(
                "🤔 ¿Qué te gustaría aprender?",
                placeholder="Escribe tu pregunta sobre el documento...",
                height=100
            )
            
            col_a, col_b = st.columns([1, 4])
            with col_a:
                ask_button = st.button("🚀 Preguntar", type="primary")
            
            if ask_button and question:
                with st.spinner("🤖 Generando explicación personalizada..."):
                    try:
                        # Usar diccionario con las 3 variables
                        enhanced_question = f"Nivel de conocimiento del estudiante: {knowledge_level}. Pregunta: {question}"
                        response = st.session_state.qa_chain.run(enhanced_question)
                        
                        st.session_state.chat_history.append((question, response))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al generar respuesta: {str(e)}")
        
        elif not api_key:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning("⚠️ Configura tu API Key para comenzar")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif not st.session_state.document_processed:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.info("📄 Sube un documento para comenzar")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")

if __name__ == "__main__": 
    main()
