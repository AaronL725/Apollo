# 易速鲜花内部文档问答系统
# 基于LangChain框架的本地文档智能问答系统，支持PDF、Word和TXT文件的智能问答

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("请设置XAI_API_KEY环境变量")

# 1.Load导入Document Loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# 加载Documents
base_dir = 'Doc_Q&A_Sys/OneFlower'  # 文件路径
documents=[]
for file in os.listdir(base_dir):
    # 构建完整的文件路径
    file_path=os.path.join(base_dir,file)
    if file.endswith('.pdf'):
        loader=PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        loader=Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader=TextLoader(file_path)
        documents.extend(loader.load())

# 2.Split将Documents切分成块以便后续进行嵌入和向量存储
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=10)
chunked_documents=text_splitter.split_documents(documents)

# 3.Store将分割嵌入并存储在矢量数据库Qdrant中
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

# 使用sentence-transformers的中文模型
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

vectorstore=Qdrant.from_documents(
    documents=chunked_documents, # 以分块的文档
    embedding=embeddings, # 用HuggingFace的Embedding模型做嵌入
    location=":memory:", # in-memory存储
    collection_name="my_documents" # 指定collection_name
)

# 4.Retrieval准备模型和Retrieval链
import logging # 导入Logging工具
from langchain_xai import ChatXAI
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain # RetrievalQA链

# 设置Logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO) # 实例化一个大模型工具-XAI的Grok
llm = ChatXAI(model_name='grok-2-latest',temperature=0)

# 实例化一个MultiQueryRetriever
retriever_from_llm=MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),llm=llm)
# 实例化一个RetrievalQA链
qa_chain=RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=retriever_from_llm)

# 5.Output问答系统的UI实现
from flask import Flask, request, render_template
app = Flask(__name__) # FlaskAPP
@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        # 接收用户输入作为问题
        question=request.form.get('question')
        print(f"\n用户问题: {question}")
        # RetrievalQA链-读入问题，生成答案
        result=qa_chain({"question":question})
        print(f"Debug - result keys: {result.keys()}")  # 打印结果字典的所有键
        print(f"\n系统回答: {result['answer']}")  # 改用'answer'键
        print(f"参考来源: {result['sources']}\n")
        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html',result=result)
    return render_template('index.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)
