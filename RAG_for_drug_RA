from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
os.environ["OPENAI_API_KEY"] = ''
os.environ["LANGSMITH_API_KEY"]=''
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
os.environ["COHERE_API_KEY"] = ''
# Reranker
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

loader = PyPDFDirectoryLoader(pdf_folder_path,extract_images=True)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap  = 100,
    length_function = len,
)

splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(splits,
                                   embedding = embeddings)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini')

faiss_retriever = db.as_retriever(# 검색 유형을 "유사도 점수 임계값"으로 설정합니다.
    search_type="similarity_score_threshold",
    # 검색 인자로 점수 임계값을 0.5로 지정합니다.
    search_kwargs={"score_threshold": 0.75})

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 20
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,faiss_retriever],weights=[0.5,0.5])

# 문서 재정렬 모델 설정
compressor = CohereRerank(model="rerank-multilingual-v3.0")

# 문맥 압축 검색기 설정
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

doc_retrieved = compression_retriever.invoke("안전성ㆍ유효성과 기준 및 시험방법의 심사를 위하여 제출하여야 하는 자료 중 완제의약품에 관한 것")

from langchain import hub
prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    llm, prompt
)

retrieval_chain = create_retrieval_chain(compression_retriever, combine_docs_chain)
query="안전성ㆍ유효성과 기준 및 시험방법의 심사를 위하여 제출하여야 하는 자료 중 완제의약품에 관한 것"
retrieval_chain.invoke({"input": query})


## 문서 검색
# 문서 재정렬 모델 설정
compressor = CohereRerank(model="rerank-multilingual-v3.0")

# 문맥 압축 검색기 설정
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# 압축된 문서 검색
compressed_docs = compression_retriever.invoke("안전성ㆍ유효성과 기준 및 시험방법의 심사를 위하여 제출하여야 하는 자료 중 완제의약품에 관한 것")

# 압축된 문서 출력
pretty_print_docs(compressed_docs)

# chain
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)
chain.invoke("안전성ㆍ유효성과 기준 및 시험방법의 심사를 위하여 제출하여야 하는 자료안전성ㆍ유효성과 기준 및 시험방법의 심사를 위하여 제출하여야 하는 자료 중 완제의약품에 관한 것")
