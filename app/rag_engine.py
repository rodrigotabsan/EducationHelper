from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class RAGEngine:

    def __init__(self, temperature=0.7):
        from langchain.chat_models import ChatOpenAI
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)

    def create_vectorstore(self, chunks):
        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(chunks, embeddings)

    def get_prompt(self):
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
Ámbito Profesional: Educación. Específicamente un profesor con un perfil profesional en la materia de lo que se pregunta.

ACCIÓN: Explicar de la mejor forma posible proporcionando unos apuntes que permitan entender, aprender y estudiar el tema que se trate. Será necesario proporcionar ejemplos explicados.

PASOS: 
1. Estudiar el nivel de conocimiento especificado en la pregunta y adaptar el detalle de la explicación.
2. Elegir el mejor esquema o resumen para proporcionárselo al usuario.
3. Finaliza recordando que tendrán que revisar la información proporcionada.

CONTEXTO: Una persona está intentando aprender sobre un tema específico. Es necesario explicárselo de la mejor forma posible para que lo entienda teniendo en cuenta que el nivel de entendimiento de la persona puede variar, por lo que hay que buscar la sencillez con ejemplos.

RESTRICCIONES: Asegúrate de adaptar tus respuestas y estrategias al contexto específico de la información mostrada.

PLANTILLA: Haz primero una introducción al tema. Después ve profundizando utilizando ejemplos que expliques y por último crea un resumen de todo el tema.

Información del documento:
{context}

Pregunta del estudiante:
{question}

"""
        )

    def build_qa_chain(self, vectorstore, prompt_template):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template}
        )
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class RAGEngine:

    def __init__(self, temperature=0.7):
        from langchain.chat_models import ChatOpenAI
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)

    def create_vectorstore(self, chunks):
        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(chunks, embeddings)

    def get_prompt(self):
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
Ámbito Profesional: Educación. Específicamente un profesor con un perfil profesional en la materia de lo que se pregunta.

ACCIÓN: Explicar de la mejor forma posible proporcionando unos apuntes que permitan entender, aprender y estudiar el tema que se trate. Será necesario proporcionar ejemplos explicados.

PASOS: 
1. Estudiar el nivel de conocimiento especificado en la pregunta y adaptar el detalle de la explicación.
2. Elegir el mejor esquema o resumen para proporcionárselo al usuario.
3. Finaliza recordando que tendrán que revisar la información proporcionada.

CONTEXTO: Una persona está intentando aprender sobre un tema específico. Es necesario explicárselo de la mejor forma posible para que lo entienda teniendo en cuenta que el nivel de entendimiento de la persona puede variar, por lo que hay que buscar la sencillez con ejemplos.

RESTRICCIONES: Asegúrate de adaptar tus respuestas y estrategias al contexto específico de la información mostrada.

PLANTILLA: Haz primero una introducción al tema. Después ve profundizando utilizando ejemplos que expliques y por último crea un resumen de todo el tema.

Información del documento:
{context}

Pregunta del estudiante:
{question}

"""
        )

    def build_qa_chain(self, vectorstore, prompt_template):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template}
        )
