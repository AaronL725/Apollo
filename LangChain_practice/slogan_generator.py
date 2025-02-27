# 情人节玫瑰宣传语（Grok）

from langchain_xai import ChatXAI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 创建LangChain的OpenAI实例
llm = ChatXAI(
    model_name="grok-2-latest",  # 模型名称
    max_tokens=200,
    temperature=0.0,
    api_key=os.getenv("XAI_API_KEY"),
    model_kwargs={"stream": True},
    verbose=False
)

# 使用LangChain调用Grok API生成文本
prompt = "请给我写一句情人节红玫瑰的中文宣传语"

# 使用流式输出
for chunk in llm.stream(prompt):
    print(chunk.content, end="", flush=True)
