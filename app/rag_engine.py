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
            template="""..."""  # Aquí va tu prompt completo como lo tienes
        )

    def build_qa_chain(self, vectorstore, prompt_template):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template}
        )
