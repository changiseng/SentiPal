import streamlit as st
from langchain.agents import Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from PIL import Image
import io
import google.generativeai as genai
import os
from semantic_router.layer import RouteLayer
from semantic_router import Route
from semantic_router.encoders import CohereEncoder, OpenAIEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import PIL.Image


os.environ["OPENAI_API_KEY"] = 
encoder = OpenAIEncoder()

playriddles = Route(
    name="riddles",
    utterances=[
        "I want to play a riddle game, give me one!",
        "Can you give me a riddle? I’ll try to guess it!",
        "I’m ready, let’s start the riddle game!"
    ],
)




# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
playuploadimage = Route(
    name="uploadimage",
    utterances=[
        "I want to upload a photo and review its details.",
        "Can I upload an image and have you review this photo for me?",
        "I have a picture to upload; could you take a look at it?",
        "I want to check the picture I uploaded; could you analyze it for me?",
        "Please let me upload the image and review its contents."
    ],
)

timeremiddrug = Route(
    name="drugremider",
    utterances=[
        "Please remind me to take my medication.",
        "When should I take my medication?",
        "I need to set a medication reminder, can you help me with that?",
        "My medication is three times a day, please help me set the reminder times.",
        "Remind me to take my medication on time, don't forget to notify me."
    ],
)


# we place both of our decisions together into single list
routes = [playriddles, playuploadimage,timeremiddrug]

rl = RouteLayer(encoder=encoder, routes=routes)

# 設定 API 金鑰
os.environ["GOOGLE_API_KEY"] = ''
genai.configure(api_key = '')


# 初始化模型
model = genai.GenerativeModel("gemini-1.5-flash")
# 假設這是我們的LLM (使用OpenAI只是佔位, 實務上請填入自己的API Key或換自己的LLM)
# 在本示範程式中，LLM並不真正call openAI，而是做為一個dummy的llm使用。
# 如果需要執行，需要有OPENAI_API_KEY環境變數或改用其他LLM.
# 如無法使用OpenAI，您可以改用一個FakeLLM，或直接改寫呼叫邏輯。

class geminillm:
    def __init__(self):
        pass
    
    def __call__(self, prompt):
        # 這裡可以設計簡單邏輯回傳假回應
        # 我們使用非常簡單的邏輯: 若prompt中出現 "玩猜謎"回服務三，
        # 若prompt中出現"上傳圖片"或 "圖片" 就回服務二，
        # 否則回服務一，若真的無法回答，就回"我不知道"
        
        lower_prompt = prompt.lower()
        router_service_name = rl(lower_prompt).name
        # 若 prompt 中包含「猜謎」或「玩猜謎」，回傳 "服務三"
        if router_service_name == 'riddles':
            return "服務三"
        # 若 prompt 中包含「圖片」或「上傳圖片」，回傳 "服務二"
        elif router_service_name == 'uploadimage':
            return "服務二"
        elif router_service_name == 'drugremider':
            return "服務一"
        # 否則，根據是否能回答問題選擇回應
        else:
            # 假設是一般聊天或是問問題
            # 使用Google的Generative AI模型來生成回應
                response = model.generate_content(lower_prompt)
                return response.text  # 返回生成的結果



# 初始化 memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 聊天記錄（用於顯示在界面上）
if "messages" not in st.session_state:
    st.session_state.messages = []

# 用來顯示圖片的列表（上傳後會顯示在對話框）
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []



st.title("SentiPal")


def analyze_image(image_path: str) -> str:
    """
    假設這個函數會將圖片進行分析，並返回圖片的描述。
    這裡我們假設有一個圖像識別的API，並簡單模擬一個返回結果。
    實際應用中，可以使用像Google Vision API，AWS Rekognition等來處理。
    """
    organ = PIL.Image.open(image_path)
    response = model.generate_content(["Analyze the image and provide a description of it.", organ])
    # 這只是模擬，實際應該連接API來進行圖像識別
    # 假設圖片描述返回如下：
    return response.text


def service_two_tool(image_path: str) -> str:
    # 假設你已經有圖片上傳並用OpenAI的API進行分析，這裡我們使用圖片識別工具(如Google Vision API)來提取圖片描述
    # 以下為虛擬的圖片識別結果，實際情況需用API進行圖片分析
    image_description = analyze_image(image_path)  # 使用外部API來分析圖片，這是示範，應根據實際API來實作

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "假設這是一張描述家庭成員在某個重要場合的照片。請根據這張圖片生成一段回憶錄，描述照片中的人物、他們的情感、場景等，並加入家庭的背景故事。以下是圖片描述： {image_description}，請生成一段溫馨且感人的家人回憶錄，將這段回憶錄以家庭故事的形式呈現，讓使用者感受到家的溫暖。使用英文回答，並依據故事內容進行出題給使用者進行問答"
            ),
            ("human", "回復家庭成員過往記憶"),
        ]
    )


    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1)

    chain = prompt | llm
    ai_msg = chain.invoke({'image_description':image_description})

    riddle_response = ai_msg.content
    # 取得生成的回憶錄內容

    return f"服務二：家人回憶錄生成 -> {riddle_response}"


def handle_service_two(image_path):
    """處理服務二的請求，生成回憶錄"""
    if "image_processed" not in st.session_state or not st.session_state.image_processed:
        memory = service_two_tool(image_path)
        # 將生成的回憶錄加入聊天記錄
        st.session_state.messages.append({"role": "assistant", "content": memory})
        st.session_state.image_processed = True  # 標記圖片已處理
    else:
        print("圖片已經處理過，不再重複處理")


# 側邊欄上傳圖片
uploaded_file = st.sidebar.file_uploader("upload a image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # 將上傳的圖片存入session，並同時顯示在對話框（在後續使用者按送出後觸發）
    image_bytes = uploaded_file.read()
    st.session_state.uploaded_images.append(image_bytes)

    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image")
    handle_service_two(io.BytesIO(image_bytes))
# 初始化 LLM（此處用FakeLLM代替）
llm = geminillm()

# 簡易的工具(未來可擴充)
def service_one_tool(query: str) -> str:

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "根據使用者提供的時間{time}及藥物{drug}進行提醒"
            ),
            ("human", "{query}"),
        ]
    )


    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1)

    chain = prompt | llm
    
    ai_msg = chain.invoke({'query':query,'time':'2024/12/16','drug':'Reminyl'})

    remind_response = ai_msg.content

    return f"服務一：生活提醒 : {remind_response}"





def service_three_tool(query: str) -> str:
    # 模擬服務三的API呼叫：猜謎遊戲

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "請生成一個有趣的謎語。謎語要能夠激發老年人的思考，適合用來進行老年人思維訓練。先不要告訴我答案",
            ),
            ("human", "{query}"),
        ]
    )


    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1)
    
    chain = prompt | llm
    ai_msg = chain.invoke({'query':query})

    riddle_response = ai_msg.content

    return f"服務三：開始猜謎遊戲！問題是：{riddle_response}"


    # 模擬服務三的API呼叫：猜謎遊戲

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system","{memory}"
               
            ),
            ("human", "{query}"),
        ]
    )


    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1)
    
    chain = prompt | llm
    ai_msg = chain.invoke({'memory':st.session_state.messages,'query':query})

    riddle_response = ai_msg.content

    return f"服務三：開始猜謎遊戲！問題是：{riddle_response}"

# 在這裡做簡易的Agent邏輯：基於使用者輸入來選擇服務
def agent_respond(user_input: str, history: list):
    # 這裡簡化：直接呼叫LLM的判斷結果
    print(user_input)
    
    # Ensure llm is a valid callable function (e.g., model invocation)
    service_decision = llm(user_input)  # Assuming llm is initialized earlier
    print(service_decision)

    # 根據判斷結果決定呼叫哪個服務
    if service_decision == "服務一":
        response = service_one_tool(user_input)
    elif service_decision == "服務二":
        # 當呼叫服務二時，顯示上傳的圖片(如果有)
        # 並把圖片描述以及服務回覆顯示出來
        response = service_two_tool(user_input)
    elif service_decision == "服務三":
        response = service_three_tool(user_input)
    else:
        # If you need to use a new model in the else block, use a different variable name.
        new_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=1
        )
        ai_msg = new_llm.invoke(history)
        response = ai_msg.content
    
    return response, service_decision


# 顯示歷史訊息(使用st.chat_message)

for msg in st.session_state.messages:
    # agent的訊息
    with st.chat_message("assistant"):
        # 如果是服務二且有上傳圖片，顯示圖片
        # 由於我們並沒有嚴格將"服務二"的回覆綁定在該訊息上，
        # 我們只能在這裡簡單判定訊息中是否包含"服務二"關鍵字
        st.write(msg["content"])                

# 使用者輸入區域
user_input = st.chat_input("請輸入你的訊息...")
if user_input:
    # 儲存使用者訊息
    st.session_state.messages.append({"role":"user", "content": user_input})

    # 對使用者輸入做agent回應
    response, service_name = agent_respond(user_input,st.session_state.messages)
    # 儲存agent回應
    st.session_state.messages.append({"role":"assistant", "content": response})
   

    # 重新刷新頁面顯示
    st.rerun()
print(st.session_state.messages)