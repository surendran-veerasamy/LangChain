# !pip3 install jupyterlab
# !pip install openai
# !pip install langchain
# !pip install python-dotenv
# !pip install pypdf 
# !pip install -U langchain-community
# !pip install chromadb
# !pip install -U langchain-openai




import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

os.environ['LANGSMITH_API_KEY']

os.environ['OPENAI_API_KEY']



from langchain_community.document_loaders import PyPDFLoader
# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("Rumesh CV-Updated 25-03-2025.pdf"),
    PyPDFLoader("Surendran_Resume_May25 v2-1.pdf"),
    PyPDFLoader("Surendran_Resume_May25 v2-1.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

#Embedding
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API_KEY'])

from langchain.vectorstores import Chroma
persist_directory = './chroma/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

# question = 'how many people\'s resume are there?'

# vectordb.similarity_search(question, k=2)

# vectordb.max_marginal_relevance_search(question,k=2, fetch_k=3)


###############################################################################################
# Build prompt
# Studying Purpose
from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

#QA
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
llm.predict("Hello world!")
# Run chain
from langchain.chains import RetrievalQA
question = "what is this document about?"
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(search_type="mmr",
                                            search_kwargs={"k": 3, "lambda_mult": 0.5}),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


result = qa_chain({"query": question})
result["result"]
##################################################################################################


#conversational
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)


question = "Who's resume are there?"
result = qa({"question": question})
result['answer']

question = "But there should be 2 people resume, right?"
result = qa({"question": question})
result['answer']
