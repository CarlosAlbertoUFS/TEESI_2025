import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub

# Configuração da página
st.set_page_config(
    page_title="Chat ENEM",
    page_icon="📚",
    layout="wide"
)

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

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

# Sidebar para upload de documentos
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
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.processed_files = True
                    st.success("Documentos processados com sucesso!")
                else:
                    st.error("Não foi possível processar os documentos.")

# Área principal para o chat
if st.session_state.processed_files:
    # Container para o histórico do chat
    chat_container = st.container()
    
    # Mostrar histórico do chat
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, dict):
                role = "user" if "human" in message else "assistant"
                content = message.get("human", message.get("ai", ""))
            else:
                continue
                
            with st.chat_message(role):
                st.write(content)
    
    # Campo de entrada para a pergunta
    user_question = st.chat_input("Digite sua pergunta sobre os documentos")
    
    if user_question:
        # Adicionar pergunta ao histórico
        st.session_state.chat_history.append({"human": user_question})
        
        # Mostrar pergunta
        with st.chat_message("user"):
            st.write(user_question)
        
        # Gerar e mostrar resposta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = st.session_state.conversation({'question': user_question})
                st.write(response['answer'])
                st.session_state.chat_history.append({"ai": response['answer']})

else:
    st.info("👈 Por favor, faça upload e processe seus documentos para começar a conversar.")