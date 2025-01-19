import streamlit as st
from dotenv import load_dotenv
from st_function import print_messages, add_message
from langchain_core.prompts import load_prompt
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# API key불러오기
load_dotenv()

# 현재 페이지 이름 설정
current_page = "BASELINE Chatgpt"

# 상태 초기화 로직
if "current_page" not in st.session_state:
    st.session_state["current_page"] = current_page

if st.session_state["current_page"] != current_page:
    # 페이지 변경 시 모든 상태 초기화
    st.session_state.clear()
    st.session_state["current_page"] = current_page

# 챗봇이름 설정
st.title("BASELINE Chatgpt")

# 대화내용을 저장하기 위한 용도로 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 사이드 바 생성
with st.sidebar:
    # 초기화버튼 생성
    clear_btn = st.button("대화내용 초기화")

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )

    # 시스템 프롬프트 추가
    user_text_prompt = st.text_area(
        "프롬프트",
        "당신은 암진단의 20년 경력을 가진 전문 AI 어시스턴스 입니다. 친절하고 공감을 형성하며 답변하세요.",
        height=150,
    )

    user_text_apply_btn = st.button("적용", key="apply")


# 체인생성
def create_chain(temperature, model_name="gpt-4o"):
    loaded_prompt = load_prompt("./prompts/counseling.yaml", encoding="utf8")
    prompt_template = user_text_prompt + loaded_prompt.template
    prompt = PromptTemplate.from_template(template=prompt_template)

    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    chain = prompt | llm | StrOutputParser()

    return chain


# 적용버튼이 눌렸을 때
if user_text_apply_btn:
    chain = create_chain(temperature=0, model_name=selected_model)
    st.session_state["chain"] = chain
    st.markdown(f"✅ 해당 프롬프트가 적용되었습니다")


# 초기화버튼이 눌리면..
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 경고 메세지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면..
if user_input:
    chain = st.session_state.get("chain", None)
    if chain is None:
        warning_msg.error("적용버튼을 눌러주세요.")

    else:
        # chain을 생성
        chain = st.session_state["chain"]

        if chain is not None:
            # 사용자의 입력
            st.chat_message("user").write(user_input)

            # 스트리밍 호출
            response = chain.stream({"question": user_input})
            with st.chat_message("assistant"):
                # 빈 공간을 만들어서, 여기에 토큰을 스트리밍 출력한다.
                container = st.empty()

                ai_answer = ""

                for token in response:
                    ai_answer += token
                    container.markdown(ai_answer)

            # 대화기록은 저장한다.
            add_message("user", user_input)
            add_message("assistant", ai_answer)
