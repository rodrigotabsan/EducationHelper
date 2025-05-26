from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class RAGEngine:

    def __init__(self, temperature=0.3):
        self.llm = OpenAI(temperature=temperature)

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)

    def create_vectorstore(self, chunks):
        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(chunks, embeddings)

    def get_prompt(self):
        return PromptTemplate(
            input_variables=["context", "question", "knowledge_level"],
            template="""
Ámbito Profesional: Educación. Específicamente un profesor con un perfil profesional en la materia de lo que se pregunta.

ACCIÓN: Explicar de la mejor forma posible proporcionando unos apuntes que permitan entender, aprender y estudiar el tema que se trate. Será necesario proporcionar ejemplos explicados.

PASOS: 
1. Estudiar según el nivel que tiene el usuario que pregunta ({knowledge_level}), el nivel de detalle de la explicación. Si es principiante no se le podrá proporcionar de primeras detalles muy muy profundos, sin embargo según avance su nivel de conocimiento se le ira introduciendo más en el tema.
2. Elegir el mejor esquema o resumen para proporcionarselo al usuario.
3. Finaliza recordando que tendrán que revisar la información proporcionada. El estudio no finaliza con los apuntes y explicaciones que das, ahora tienen que buscar información de una fuente más fiable para revisar que los datos proporcionados son acertados.

CONTEXTO: Una persona está intentando aprender sobre un tema específico. Es necesario explicárselo de la mejor forma posible para que lo entienda teniendo en cuenta que el nivel de entendimiento de la persona puede variar, por lo que hay que buscar la sencillez con ejemplos.

RESTRICCIONES: Asegúrate de adaptar tus respuestas y estrategias al contexto específico de la información mostrada.

PLANTILLA: Haz primero una introducción al tema. Después ve profundizando utilizando ejemplos que expliques y por último crea un resumen de todo el tema.

Información del documento:
{context}

Pregunta del estudiante:
{question}

Respuesta adaptada al nivel {knowledge_level}:
"""
        )

    def build_qa_chain(self, vectorstore, prompt_template):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template}
        )
