import os
import chromadb
from chromadb.utils import embedding_functions
import shutil
import re
from tqdm import tqdm

def create_vector_database():
    # 定义路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    unextracted_files_dir = os.path.join(current_dir, "unextracted_files")
    vectorized_data_dir = os.path.join(current_dir, "vectorized_data")
    
    # 确保向量数据库目录存在，如果已存在则先删除
    if os.path.exists(vectorized_data_dir):
        shutil.rmtree(vectorized_data_dir)
    os.makedirs(vectorized_data_dir, exist_ok=True)
    
    # 创建持久化的Chroma客户端
    client = chromadb.PersistentClient(path=vectorized_data_dir)
    
    # 使用默认的句子转换器嵌入函数
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
    
    # 获取所有.md文件
    md_files = [f for f in os.listdir(unextracted_files_dir) if f.endswith('.md')]
    
    # 为每个文件创建独立的集合
    total_files = len(md_files)
    for idx, md_file in enumerate(tqdm(md_files), 1):
        print(f"处理文件 {idx}/{total_files}: {md_file}")
        # 使用文件名创建集合（移除.md后缀）
        collection_name = os.path.splitext(md_file)[0]
        
        # 创建集合
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,  # 提高索引质量
                "hnsw:search_ef": 200  # 提高搜索质量
            }
        )
        
        # 读取文件内容
        file_path = os.path.join(unextracted_files_dir, md_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, 'r', encoding='gbk') as file:
                content = file.read()
        
        # 将内容分块，每个块最大1000个字符（可以根据需要调整）
        chunks = split_into_chunks(content, max_length=1000)
        
        # 为每个块创建唯一ID
        ids = [f"{collection_name}_chunk_{i}" for i in range(len(chunks))]
        
        # 添加到集合中
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=[{"source": md_file, "chunk_index": i} for i in range(len(chunks))]
        )
        
        print(f"已将文件 {md_file} 添加到向量数据库中，共 {len(chunks)} 个块")
    
    print(f"向量数据库已成功创建，存储在 {vectorized_data_dir} 目录中")
    return client

def split_into_chunks(text, max_length=800, overlap=80):
    """
    将文本分割成重叠的块
    """
    # 尝试先按帖子分割
    posts = []
    post_markers = re.findall(r'\* \[#\d+\].*?(?=\* \[#\d+\]|\Z)', text, re.DOTALL)
    
    if post_markers and len(post_markers) > 2:
        # 如果能识别出多个帖子，先按帖子处理
        for post in post_markers:
            # 主要帖子(较长内容)进一步分块
            if len(post) > max_length:
                # 对长帖子使用常规分块方法
                post_chunks = _split_text_by_semantic(post, max_length, overlap)
                posts.extend(post_chunks)
            else:
                # 短帖子保持完整
                posts.append(post)
        return posts
    else:
        # 无法按帖子分割，使用语义分块
        return _split_text_by_semantic(text, max_length, overlap)

def _split_text_by_semantic(text, max_length=800, overlap=80):
    """
    按语义分割文本
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_length, len(text))
        
        # 优先在帖子边界分割
        if end < len(text):
            post_boundary = text.rfind("* [#", start, end)
            if post_boundary > start + max_length // 2:
                end = post_boundary
            else:
                # 其次在段落边界分割
                paragraph_boundary = text.rfind("\n\n", start, end)
                if paragraph_boundary > start + max_length // 2:
                    end = paragraph_boundary + 2
                else:
                    # 再次在列表项边界分割
                    list_boundary = max(
                        text.rfind("\n- ", start, end),
                        text.rfind("\ni - ", start, end),
                        text.rfind("\nii- ", start, end)
                    )
                    if list_boundary > start + max_length // 2:
                        end = list_boundary
                    else:
                        # 最后在句子边界分割
                        sentence_boundary = max(
                            text.rfind(". ", start, end),
                            text.rfind("? ", start, end),
                            text.rfind("! ", start, end)
                        )
                        if sentence_boundary > start + max_length // 2:
                            end = sentence_boundary + 2
                        else:
                            # 如果都找不到，在空白处分割
                            while end > start + max_length - overlap and not text[end].isspace():
                                end -= 1
                            
                            if end <= start + max_length - overlap:
                                end = start + max_length
        
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    
    return chunks

def query_document(client, document_name, query_text, n_results=5, where_filter=None):
    """
    查询特定文档的内容
    """
    # 移除可能的.md后缀
    collection_name = document_name
    if collection_name.endswith('.md'):
        collection_name = os.path.splitext(collection_name)[0]
    
    # 获取集合
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        print(f"未找到文档 {document_name} 的集合")
        return None
    
    # 执行查询
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter  # 添加过滤条件
    )
    
    return results

if __name__ == "__main__":
    # 创建向量数据库
    client = create_vector_database()
    
    # 移除查询示例代码
    print("\n向量数据库创建完成，可以通过query_document函数进行查询")
    print("示例: query_document(client, '文档名', '查询文本')")
