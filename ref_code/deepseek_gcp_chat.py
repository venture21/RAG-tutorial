from pathlib import Path
import os
import sys
from typing import Optional
from dotenv import load_dotenv
import google.auth
import google.auth.transport.requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 환경 변수 로드 (.env)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item if isinstance(item, str) else item.get("text", "") if isinstance(item, dict) else "" for item in content)
    return ""

def get_gcp_access_token() -> str:
    """
    Google Cloud Application Default Credentials(ADC)로부터 Access Token을 발급받습니다.
    gcloud auth application-default print-access-token 과 동일한 토큰입니다.
    """
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    return credentials.token

def create_deepseek_chat_model(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    streaming: bool = True
) -> ChatOpenAI:
    """
    GCP Vertex AI Model Garden의 DeepSeek V4.1 Flash API Service를 위한 ChatOpenAI 인스턴스를 생성합니다.
    공식 가이드 규격:
    - ENDPOINT: aiplatform.googleapis.com
    - REGION: global
    - MODEL: deepseek-ai/deepseek-v4.1-flash-maas
    - API URL: https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/global/endpoints/openapi
    """
    project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-ai/deepseek-v4.1-flash-maas")

    if not project_id or project_id == "YOUR_PROJECT_ID":
        raise ValueError("GOOGLE_CLOUD_PROJECT 환경 변수 또는 project_id 인자가 필요합니다.")

    # 공식 가이드에 맞춘 엔드포인트 URL 구성 (REGION=global인 경우 aiplatform.googleapis.com 사용)
    if location == "global":
        endpoint_host = "aiplatform.googleapis.com"
    else:
        endpoint_host = f"{location}-aiplatform.googleapis.com"

    # LangChain ChatOpenAI가 뒤에 /chat/completions를 붙이므로 openapi까지를 base_url로 지정
    base_url = f"https://{endpoint_host}/v1/projects/{project_id}/locations/{location}/endpoints/openapi"
    

    print(f"\n[GCP Vertex AI DeepSeek 설정]")
    print(f"- Project ID: {project_id}")
    print(f"- Location  : {location}")
    print(f"- Model     : {model_name}")
    print(f"- Base URL  : {base_url}\n")

    # LangChain ChatOpenAI 초기화
    chat_model = ChatOpenAI(
        model=model_name,
        api_key=get_gcp_access_token,
        openai_api_base=base_url,
        temperature=temperature,
        streaming=streaming,
    )
    return chat_model

def run_cli_chatbot():
    """
    LangChain 기반 DeepSeek CLI 챗봇 실행 루프
    """
    try:
        llm = create_deepseek_chat_model()
    except Exception as e:
        print(f"❌ LLM 초기화 실패: {e}")
        print("\n💡 팁: 'gcloud auth application-default login'을 실행하여 GCP 인증을 완료했는지 확인하세요.")
        return

    system_prompt = "You are a helpful, accurate, and friendly AI assistant powered by DeepSeek V4.1 Flash on Google Cloud Vertex AI."
    history = [SystemMessage(content=system_prompt)]

    print("=" * 60)
    print("🤖 GCP Vertex AI DeepSeek V4.1 Flash 챗봇에 오신 것을 환영합니다!")
    print("   (종료: 'exit', 'quit', 'q' | 대화 초기화: 'clear')")
    print("=" * 60)

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

            print("🤖 Assistant: ", end="", flush=True)

            # HTTP 클라이언트가 토큰 공급 콜백으로 최신 ADC 토큰을 사용합니다.
            # 스트리밍 응답
            full_response = ""
            for chunk in llm.stream(history):
                content = extract_text(chunk.content)
                if content:
                    print(content, end="", flush=True)
                    full_response += content
            
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
