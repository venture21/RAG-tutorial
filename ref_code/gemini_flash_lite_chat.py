from pathlib import Path
import os
import sys
import warnings
from typing import Optional, Any
from dotenv import load_dotenv
import google.auth
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 환경 변수 로드 (.env)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# 고정 샘플링 경고 등 불필요한 UserWarning 숨김 처리
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

def extract_text(content: Any) -> str:
    """
    LangChain chunk.content가 str이거나 list[dict] 형태일 때 텍스트만 안전하게 추출합니다.
    예: [{'type': 'text', 'text': 'Hello', 'index': 0}] -> 'Hello'
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])
        return "".join(text_parts)
    return str(content) if content else ""

def create_gemini_flash_lite_model(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model_name: Optional[str] = None,
    streaming: bool = True
) -> ChatGoogleGenerativeAI:
    """
    GCP Vertex AI 기반 Gemini 3.5 Flash-Lite 모델용 ChatGoogleGenerativeAI 인스턴스를 생성합니다.
    """
    credentials, default_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
    location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    model_name = model_name or os.getenv("GEMINI_FLASH_LITE_MODEL", "gemini-3.5-flash-lite")

    if not project_id or project_id == "YOUR_PROJECT_ID":
        raise ValueError("Google Cloud 프로젝트 ID가 설정되지 않았습니다. GOOGLE_CLOUD_PROJECT 환경 변수를 확인하세요.")

    print(f"\n[GCP Vertex AI Gemini 설정]")
    print(f"- Project ID : {project_id}")
    print(f"- Location   : {location}")
    print(f"- Model      : {model_name}\n")

    # Vertex AI Gemini 모델 초기화 (Fixed sampling 모델이므로 temperature 제외)
    chat_model = ChatGoogleGenerativeAI(
        model=model_name,
        vertexai=True,
        api_key=None,
        credentials=credentials,
        project=project_id,
        location=location,
        streaming=streaming,
        timeout=120,
        max_retries=1,
    )
    return chat_model

def run_cli_chatbot():
    """
    LangChain 기반 Gemini 3.5 Flash-Lite 대화형 챗봇
    """
    try:
        llm = create_gemini_flash_lite_model()
    except Exception as e:
        print(f"❌ Gemini 모델 초기화 실패: {e}")
        print("\n💡 팁: 'gcloud auth application-default login'을 실행하여 GCP 인증을 완료했는지 확인하세요.")
        return

    system_prompt = "You are a helpful, fast, and polite AI assistant powered by Gemini 3.5 Flash-Lite on Google Cloud Vertex AI."
    history = [SystemMessage(content=system_prompt)]

    print("=" * 65)
    print("⚡ GCP Vertex AI Gemini 3.5 Flash-Lite 챗봇에 오신 것을 환영합니다!")
    print("   - 종료: 'exit', 'quit', 'q'")
    print("   - 대화 기록 초기화: 'clear'")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n👤 User: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 챗봇을 종료합니다. 좋은 하루 되세요!")
                break

            if user_input.lower() == "clear":
                history = [SystemMessage(content=system_prompt)]
                print("\n🧹 대화 기록이 초기화되었습니다.")
                continue

            # 사용자 메시지 히스토리에 추가
            history.append(HumanMessage(content=user_input))

            print("⚡ Assistant: ", end="", flush=True)

            # 스트리밍 응답 출력
            full_response = ""
            for chunk in llm.stream(history):
                text_chunk = extract_text(chunk.content)
                if text_chunk:
                    print(text_chunk, end="", flush=True)
                    full_response += text_chunk
            
            print()

            # 어시스턴트 메시지 히스토리에 추가
            history.append(AIMessage(content=full_response))

        except KeyboardInterrupt:
            print("\n\n👋 대화가 중단되었습니다. 챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    run_cli_chatbot()
