import os
import chromadb
from chromadb.utils import embedding_functions
import shutil
import re
from tqdm import tqdm

def create_vector_database():
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    unextracted_files_dir = os.path.join(current_dir, "unextracted_files")
    vectorized_data_dir = os.path.join(current_dir, "vectorized_data")
    
    # 清理并重建向量数据库目录
    if os.path.exists(vectorized_data_dir):
        shutil.rmtree(vectorized_data_dir)
    os.makedirs(vectorized_data_dir, exist_ok=True)
    
    # 初始化持久化的ChromaDB客户端
    client = chromadb.PersistentClient(path=vectorized_data_dir)
    
    # 设置句子转换器作为嵌入函数
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
    
    # 获取待处理的markdown文件列表
    md_files = [f for f in os.listdir(unextracted_files_dir) if f.endswith('.md')]
    
    # 遍历处理每个文件
    total_files = len(md_files)
    for idx, md_file in enumerate(tqdm(md_files), 1):
        print(f"处理文件 {idx}/{total_files}: {md_file}")
        # 从文件名创建集合名称
        collection_name = os.path.splitext(md_file)[0]
        
        # 创建向量集合，设置HNSW索引参数
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,  # 提高索引质量的参数
                "hnsw:search_ef": 200  # 提高搜索质量的参数
            }
        )
        
        # 读取文件内容，自动处理编码问题
        file_path = os.path.join(unextracted_files_dir, md_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            # 尝试备选编码
            with open(file_path, 'r', encoding='gbk') as file:
                content = file.read()
        
        # 将内容分块处理
        chunks = split_into_chunks(content, max_length=1000)
        
        # 为每个块生成唯一ID
        ids = [f"{collection_name}_chunk_{i}" for i in range(len(chunks))]
        
        # 将文档块添加到向量数据库
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=[{"source": md_file, "chunk_index": i} for i in range(len(chunks))]
        )
        
        print(f"已将文件 {md_file} 添加到向量数据库中，共 {len(chunks)} 个块")
    
    print(f"向量数据库已成功创建，存储在 {vectorized_data_dir} 目录中")
    return client

def split_into_chunks(text, max_length=800, overlap=80):
    # 智能文本分块函数，优先按帖子结构分割
    posts = []
    post_markers = re.findall(r'\* \[#\d+\].*?(?=\* \[#\d+\]|\Z)', text, re.DOTALL)
    
    if post_markers and len(post_markers) > 2:
        # 存在多个帖子标记时，按帖子进行分割
        for post in post_markers:
            # 对长帖子进行进一步分块
            if len(post) > max_length:
                post_chunks = _split_text_by_semantic(post, max_length, overlap)
                posts.extend(post_chunks)
            else:
                # 短帖子保持完整
                posts.append(post)
        return posts
    else:
        # 无法识别帖子结构时，使用语义分块
        return _split_text_by_semantic(text, max_length, overlap)

def _split_text_by_semantic(text, max_length=800, overlap=80):
    # 按语义边界智能分割文本的内部函数
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_length, len(text))
        
        # 分割优先级：帖子边界 > 段落边界 > 列表项边界 > 句子边界 > 空白字符
        if end < len(text):
            post_boundary = text.rfind("* [#", start, end)
            if post_boundary > start + max_length // 2:
                end = post_boundary
            else:
                paragraph_boundary = text.rfind("\n\n", start, end)
                if paragraph_boundary > start + max_length // 2:
                    end = paragraph_boundary + 2
                else:
                    list_boundary = max(
                        text.rfind("\n- ", start, end),
                        text.rfind("\ni - ", start, end),
                        text.rfind("\nii- ", start, end)
                    )
                    if list_boundary > start + max_length // 2:
                        end = list_boundary
                    else:
                        sentence_boundary = max(
                            text.rfind(". ", start, end),
                            text.rfind("? ", start, end),
                            text.rfind("! ", start, end)
                        )
                        if sentence_boundary > start + max_length // 2:
                            end = sentence_boundary + 2
                        else:
                            # 无合适边界时，在空白处分割
                            while end > start + max_length - overlap and not text[end].isspace():
                                end -= 1
                            
                            if end <= start + max_length - overlap:
                                end = start + max_length
        
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    
    return chunks

def query_document(client, document_name, query_text, n_results=5, where_filter=None):
    # 查询特定文档向量集合的函数
    collection_name = document_name
    if collection_name.endswith('.md'):
        collection_name = os.path.splitext(collection_name)[0]
    
    # 尝试获取集合
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        print(f"未找到文档 {document_name} 的集合")
        return None
    
    # 执行相似度查询
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter  # 可选的过滤条件
    )
    
    return results

if __name__ == "__main__":
    # 程序入口点
    client = create_vector_database()
    
    print("\n向量数据库创建完成")
