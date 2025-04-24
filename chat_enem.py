import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import PrivateAttr
from langchain.callbacks import StreamlitCallbackHandler
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Configuração da página
st.set_page_config(
    page_title="Chat ENEM",
    page_icon="📚",
    layout="wide"
)

# Configurar a chave da API HuggingFace
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Substitua com seu token
os.environ["OPENAI_API_KEY"] = "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"  # <- this has to be your api key!

# Título
st.title("Chat ENEM 📚")
st.markdown("Faça perguntas sobre seus documentos do ENEM!")

# Inicialização das variáveis de estado da sessão
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = False

def process_file(uploaded_file):
    try:
        # Criar diretório temporário se não existir
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Salvar arquivo temporariamente
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Carregar documento
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Formato de arquivo não suportado")
        
        documents = loader.load()
        
        # Limpar arquivo temporário
        os.remove(file_path)
        return documents
    except Exception as e:
        st.error(f"Erro ao processar arquivo {uploaded_file.name}: {str(e)}")
        return []

def get_text_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)


def get_vectorstore_with_retriever(vectorstore):
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.2)

    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=vectorstore.as_retriever()
    )

class LocalT5Wrapper(LLM):
    _pipeline: any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_name = "google/mt5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        object.__setattr__(self, "_pipeline", pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device))

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = self._pipeline(prompt, max_length=512, do_sample=True, temperature=0.5)
        return result[0]["generated_text"]

    @property
    def _llm_type(self) -> str:
        return "local-t5-wrapper"
    
def get_conversation_chain(vectorstore_with_retriever):
    #llm = LocalT5Wrapper()
    #llm = init_chat_model('google/mt5-base', model_provider='huggingface')
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    # llm = HuggingFaceHub(
    #     repo_id="unicamp-dl/ptt5-base-portuguese-vocab",  # Modelo que suporta text2text-generation e é adequado ao português
    #     model_kwargs={"temperature": 0.5, "max_length": 512}  # Configurações do modelo
    # )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore_with_retriever,
        memory=memory,
        verbose=True
    )

    return conversation_chain

# Sidebar para upload de documentos
vectorstore = None
with st.sidebar:
    st.subheader("Seus documentos")
    uploaded_files = st.file_uploader(
        "Faça upload dos seus documentos em PDF ou DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    if st.button("Processar Documentos"):
        if not uploaded_files:
            st.error("Por favor, faça upload de pelo menos um documento.")
        else:
            with st.spinner("Processando documentos..."):
                # Processar todos os documentos
                all_documents = []
                for uploaded_file in uploaded_files:
                    documents = process_file(uploaded_file)
                    all_documents.extend(documents)
                
                if all_documents:
                    # Criar chunks e vetorizar
                    text_chunks = get_text_chunks(all_documents)
                    vectorstore = get_vectorstore_with_retriever(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.processed_files = True
                    st.success("Documentos processados com sucesso!")
                else:
                    st.error("Não foi possível processar os documentos.")

# Área principal para o chat
if st.session_state.processed_files:
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, dict):
                role = "user" if "human" in message else "assistant"
                content = message.get("human", message.get("ai", ""))
            else:
                continue
            with st.chat_message(role):
                st.write(content)

    user_question = st.chat_input("Digite sua pergunta sobre os documentos")

    if user_question:
        st.session_state.chat_history.append({"human": user_question})
        with st.chat_message("user"):
            st.write(user_question)
        # Gerar e mostrar resposta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = st.session_state.conversation({'question': user_question})
                answer = response.get('answer', 'Desculpe, não consegui gerar uma resposta.')  # Extrair apenas a resposta
                st.write(answer)
                st.session_state.chat_history.append({"ai": answer})

else:
    st.info("👈 Por favor, faça upload e processe seus documentos para começar a conversar.")