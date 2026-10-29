모델 단독 채팅 참고 코드
======================

ref_code 폴더를 RAG-tutorial 안으로 이동했습니다.
4개 스크립트는 RAG 검색 없이 모델과 스트리밍으로 대화하는 참고 예제입니다.
대화 기록은 프로세스 메모리에만 보관합니다.

상위 폴더의 requirements.txt로 설치하고, 상위 .env와 본인 ADC를 사용합니다.
아래 명령은 프로그램 폴더에서 실행합니다.

macOS / Linux:
.venv/bin/python -X utf8 ref_code/gemini_flash_lite_chat.py

Windows PowerShell:
.\.venv\Scripts\python.exe -X utf8 ref_code\gemini_flash_lite_chat.py

claude_haiku_chat.py, deepseek_gcp_chat.py, gemma4_gcp_chat.py도 같은 방법으로 실행합니다.
메인 서비스와 같은 의존성을 사용합니다. 모델·리전의 제공 여부와 접근 권한은 본인 프로젝트에서 확인하세요.
실제 대화는 본인 Google Cloud 프로젝트를 사용하므로 비용이 발생할 수 있습니다.
DeepSeek/Gemma는 토큰 공급 콜백으로 인증을 갱신하고 키를 코드에 넣지 않습니다.
이번 배포 검증에서는 실제 온라인 대화를 실행하지 않았습니다.
