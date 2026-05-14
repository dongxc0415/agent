'''
1. Memory = Agent对过去信息的存储 + 检索 + 使用

1.1 ConversationBufferMemory:
llm = ChatTongyi(model_name = "qwen-max")
memory = ConversationBufferMemory(return_messages = True)                    # 默认为false 返回字符串 我们需要message对象 几乎为ture
conversation = ConversationChain(llm = llm, memory = memory, verbose = True)  # verbose = ture 可以看到提示词变化
缺点: token消耗多 干扰信息多 没有着重点 存在输入限制
为了解决以上缺点 一般memory采用如下：

1.2 ConversationSummaryMemory
llm = ChatTongyi(model_name = "qwen-max")
memory = ConversationSummaryMemory(llm = llm, return_messages = True)
conversation = ConversationChain(llm = llm, memory = memory, verbose = True)
优点: 节省空间 节省token 会总结上文成为一段摘要  降噪：进行摘要的时候会处理掉一些垃圾信息
缺点： 丢失细节 进行摘要一定会损失信息 无法获取全部的原信息

1.3 ConversationSummaryBufferMemory
llm = ChatTongyi(model_name = "qwen-max")
memory = ConversationSummaryBufferMemory(llm = llm, max_token_limit = ***, return_messages = True)
conversation = ConversationChain(llm = llm, memory = memory, verbose = True )
优点; 既有摘要 又有原文
按照token的上线 max_token_limit = *** 进行切分 当这一轮新的摘要 + 原文进去conversation之后 超过了这个上限 就会把第一轮（也可能是第一轮 第二轮的
摘要 + 原文 都传给 旧摘要 memory.moving_summary_buffer里）        # print(memory.moving_summary_buffer)
旧摘要作用： 作为背景、人设偏好
缺点：可能会乱编 胡说八道 损失精度

1.4 ConversationBufferWindowMemory
llm = ChatTongyi(model_name = "qwen-max")
memory = ConversationBufferWindowMemory(
    memory_key = "chat_history",
    return_messages = True,
    k = 5,
    output_key = "output",
    ai_prefix = "助手"
)
优点：响应速度快 会记住k轮的问题

1.5 ConversationEntityMemory
llm = ChatTongyi(model_name = "qwen-max")
memory = ConversationEntityMemory(
    llm = llm ,
    k = 3,
    return_messages = True,
    entity_extraction_prompt = ENTITY_EXTRACTION_PROMPT,
    entity_summarization_prompt = ENTITY_SUMMARIZATION_PROMPT,
    chat_history_key = "history"
)
把过去的100轮的信息都保存存入"history"中 更新实体只更新最近k轮 回答只回答最近k轮的原文和摘要 过去的100轮只存储不读取
优点：
1. 实体感知 能记住用户的信息 而不是单纯记住几轮的消息
2. 长期记忆 节省token
3. 摘要

1.6 VectorStoreRetrieverMemory
llm = ChatTongyi(model_name="qwen-max")

embeddings = OpenAIEmbeddings()                                # 向量化工具
vectorstore = FAISS.load_local("my_faiss_index", embeddings)   # 已存在的索引，也可以创建新的
retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3}                                   # 每次返回最相关的 3 条历史
)


memory = VectorStoreRetrieverMemory(
    retriever = retriever,
    memory_key = "history",
    input_key = "question",
    return_docs = True,
    exclude_input_keys = ("metadata",)
)
优点：
1. 可以用于长期记忆系统
2. 通过向量化 把用户提问的问题question 和历史记录history转换为向量embedding 然后通过余弦相似度 选出来最相似的k条信息



'''


# todo 4优化
'''
1. 记忆体优化 短期记忆：保留最近5轮对话 长期记忆：用VectorStore存储过去所有对话 + 用户偏好
使用 Chroma + DashScopeEmbeddings 构建长期向量存储库
每轮对话结束后 保存用户问题 + llm回答
通过similarity_search检索最相关历史对话
将检索结果传入提示词prompt的{long_term_context} 
'''
# 导包
import requests
import os
import json
import gradio as gr
from datetime import datetime
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()


vectorstore = None


# todo 1. Tool 编写

# 1.1 日期获取
@tool
def get_current_date() -> str:
    """获取当前真实的日期和时间。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


# 1.2 数字计算
@tool
def calculator(expression: str) -> str:
    """执行数学计算。例如 '3 * (5 + 2)'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


# 1.3 天气获取 3
@tool
def get_weather(city: str) -> str:
    """查询指定城市当前的真实天气。"""

    amap_key = "d0199b506ab96d81baab3f10aff1482c"
    # amap_key = os.getenv("amap_key")
    try:
        # 获取城市的 adcode (高德API需要城市编码 例如大连： 210200)
        # 协议: // 域名 / 路径?参1 = 值1 & 参2 = 值2  参1:城市名字 参2;API Key
        geo_url = f"https://restapi.amap.com/v3/geocode/geo?address={city}&key={amap_key}"
        geo_resp = requests.get(geo_url, timeout=5)
        # json字符串 -> 字典
        geo_data = geo_resp.json()

        if geo_data["status"] == "1" and geo_data["geocodes"]:
            # 取第一个城市 取唯一的行政编码 六位数字 例如大连： 210200
            adcode = geo_data["geocodes"][0]["adcode"]

            # 查询实时天气 参1：城市位置编码 参2：api key
            weather_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={amap_key}"
            w_resp = requests.get(weather_url, timeout=5)
            w_data = w_resp.json()

            if w_data["status"] == "1" and w_data["lives"]:
                live = w_data["lives"][0]
                return f"{live['city']}当前实时天气：{live['weather']}，温度 {live['temperature']}℃，风向 {live['winddirection']}风，湿度 {live['humidity']}%。"

        return f"抱歉，没能找到 {city} 的实时天气信息。"

    except Exception as e:
        return f"网络连接失败，请检查网络设置。错误详情: {e}"


# todo 1.4 文档回答  换用Pydantic Schema                                     todo 音标：Pydantic：/paɪˈdæntɪk/   Schema：/ˈskiːmə/
# 1.4.1. 定义参数模型
class PDFSearchInput(BaseModel):
    # ***_name: type = Field(default = value, description = "给模型看的详细说明")
    query: str = Field(description = "需要从PDF文档中检索的具体问题或关键词")
    k: int = Field(default = 4, description = "返回最相关的文本段落数量，默认为 4")
    score_threshold: float = Field(default = 0.7, description = "向量余弦相似度，范围是0-1，值越接近1表示内容越相似 ")


# 1.4.2. 绑定 Schema 并定义工具逻辑
@tool(args_schema = PDFSearchInput)
def search_pdf_knowledge(query: str, k: int, score_threshold: float):
    """当用户需要总结文档、询问细节时调用。"""

    global vectorstore
    if vectorstore is None:
        return "错误：请先构建知识库。"

    # print(f"调试信息")
    # print(f"模型生成的 Query: {query}")
    # print(f"模型生成的 k: {k}")
    # print(f"模型生成的 threshold: {score_threshold}")

    # 1. 拿到原始结果（包含文档和分数）
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    # 2. 先不管 threshold，打印一下数据库返回的真实分数到底是多少
    for i, (doc, score) in enumerate(docs_and_scores):
        print(f"片段 {i} 原始得分: {score}")

    # 3. 如果结果为空，尝试降低要求再搜一次
    if not docs_and_scores:
        return "向量数据库检索返回空列表，请检查 PDF 内容是否已成功加载。"

    # 4. 拼接内容
    context = "\n\n".join([doc.page_content for doc, score in docs_and_scores])

    return f"以下是从文档中检索到的参考内容：\n\n{context}"


# 1.5 个人偏好（流动可变性）
@tool
def manage_user_preference(action: str, key: str, value: str = None) -> str:
    """
    管理用户个人偏好（如：编程风格、饮食、习惯）。
    action: 'save'(保存) 或 'get'(获取)。
    """
    file_path = "user_profile.json"
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        if action == "save":
            data[key] = value
            with open(file_path, "w", encoding="utf-8") as f:
                # 修复：移除多余的 dump
                json.dump(data, f, ensure_ascii=False, indent=4)
            return f"已记录：{key} -> {value}"
        elif action == "get":
            return data.get(key, f"未找到关于 {key} 的记录。")
    except Exception as e:
        return f"存储出错: {e}"




# todo 2. 模型封装

llm = ChatTongyi(model_name="qwen-max")

# 工具列表
tools = [get_current_date, calculator, get_weather, search_pdf_knowledge, manage_user_preference]

# 提示词Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """你是一个进化版智能助手 你需要注意以下若干点：
    1.  当用户询问“这篇论文”、“这个文档”或相关内容时，你必须主动调用 search_pdf_knowledge 工具进行检索，不要假设你已经知道了。
    2.  在回答前，若认为用户习惯可能影响答案，请先尝试获取相关偏好。
    3.  拥有短期对话记忆：能理解上下文。
    4.  拥有长期知识库：涉及文档时请主动调用 search_pdf_knowledge。
    5.  如果调用 search_pdf_knowledge 后未找到相关信息 请如实告知用户‘文档中未提及此内容’ 严禁根据自身知识库瞎编文档细节
    6.  拥有个人画像：当用户提到其喜好、习惯、特定要求时，必须使用 manage_user_preference 进行存储。
    7.  如果用户询问天气、日期、计算等，请使用相应工具
    8.  我目前住在长沙 以后我问‘今天天气怎么样’的时候 你直接默认按照长沙查就可以 不需要额外问我居住地址
    9.  对于复杂任务：在给出最终答案前 先用一句话简要说明你的处理思路 例如：‘我将先为您检索文档，再结合实时天气为您提供建议。
    10. 你可以根据用户问题的复杂度，自主调整检索条数：k 和向量余弦相似度：score_threshold
    11. 请参考检索到的长期相关记忆来辅助回答：{long_term_context}
    """),   
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 构建记忆体
memory = ConversationBufferWindowMemory(
    memory_key = "chat_history",
    return_messages = True,
    k = 5,
    output_key = "output",
    ai_prefix = "助手"
)

# 新增 长期向量vector存储库
long_term_db = Chroma(
    collection_name = "chat_log_vectors",
    embedding_function = DashScopeEmbeddings(model = "text-embedding-v1"),
    persist_directory = "./long_term_storage"
)

# 创建 Agent 执行器
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    memory = memory,
    verbose = True,
    handle_parsing_errors = True
)




# todo 3. 前端界面 (Gradio 实现)

def process_file(file):
    """处理上传的 PDF 并构建知识库"""
    global vectorstore
    if file is not None:
        try:
            loader = PyMuPDFLoader(file.name)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)

            embeddings = DashScopeEmbeddings(model="text-embedding-v1")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )
            return "✅ 文档加载成功，可以开始提问了！"
        except Exception as e:
            return f"❌ 加载失败: {e}"
    return "未检测到文件"


def predict(message, history):
    """处理对话逻辑"""
    # 1. 在执行前，先根据当前问题 message 检索长期记忆库
    past_docs = long_term_db.similarity_search(message, k=2)
    context_str = "\n".join([d.page_content for d in past_docs]) if past_docs else "无相关历史记录"

    # 2. 调用时手动传入 long_term_context 变量
    response = agent_executor.invoke({
        "input": message,
        "long_term_context": context_str
    })

    final_output = response["output"]

    # 3. 回答结束后，将本轮对话存入长期库
    long_term_db.add_texts([f"用户：{message}\n助手：{final_output}"])

    return final_output


# 构建界面
with gr.Blocks(title="池塘边上树的Agent智能助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 池塘边上树的Agent智能助手")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 文档中心")
            file_input = gr.File(label="上传 PDF 供 Agent 参考", file_types=[".pdf"])
            upload_button = gr.Button("构建知识库", variant="primary")
            status_output = gr.Textbox(label="状态", interactive=False)
            upload_button.click(process_file, inputs=[file_input], outputs=[status_output])

        with gr.Column(scale=4):
            gr.ChatInterface(fn=predict)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)




    # npm install -g @anthropic/claude-cli