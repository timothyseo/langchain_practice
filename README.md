Notes for practicing RAG using langchain


## 포트폴리오 프로젝트: 대한민국 헌법 질의응답 챗봇 (LangChain RAG 기반)

**프로젝트 개요:**

LLM(Large Language Model)의 환각(Hallucination) 현상을 최소화하고, 특정 도메인(대한민국 헌법)에 대한 정확하고 근거 있는 답변을 제공하기 위해 LangChain 프레임워크와 RAG(Retrieval-Augmented Generation) 기술을 활용하여 헌법 질의응답 챗봇 시스템을 구축하고 Streamlit을 통해 웹 애플리케이션으로 배포했습니다. 사용자는 자연어 질문을 통해 헌법의 특정 조항이나 내용에 대해 쉽고 빠르게 정보를 얻을 수 있습니다.

**목표:**

*   LangChain RAG 파이프라인 구축 및 이해도 향상
*   대한민국 헌법 텍스트를 기반으로 사실에 입각한 답변 생성 능력 확보
*   LLM의 한계점을 RAG 기술로 보완하는 실용적인 AI 애플리케이션 개발 경험
*   Streamlit을 이용한 간편한 LLM 기반 서비스 배포 경험

**핵심 기능:**

*   **헌법 기반 질의응답:** 사용자의 질문 의도를 파악하여 대한민국 헌법 원문 중 가장 관련성 높은 부분을 검색
*   **근거 제시 답변:** 검색된 헌법 내용을 바탕으로 LLM이 답변을 생성하여, 정보의 출처와 신뢰도를 높임
*   **웹 기반 인터페이스:** Streamlit을 활용하여 사용자가 편리하게 질문하고 답변을 확인할 수 있는 UI 제공

**기술 스택:**

*   **언어 모델 (LLM):** `EEVE-Korean-10.8B:latest` (Ollama를 통해 로컬 구동) - 한국어 특화 모델 활용
*   **프레임워크:** LangChain - RAG 파이프라인 구축 및 LLM 연동
*   **문서 로더 (Document Loader):** `PDFLoader` (또는 헌법 텍스트 파일 형식에 맞는 로더) - 헌법 원문 로딩
*   **텍스트 분할 (Text Splitter):** `RecursiveCharacterTextSplitter` (`chunk_size`, `chunk_overlap` 파라미터 조정) - 문맥 유지를 고려한 효율적인 텍스트 분할
*   **임베딩 모델 (Embedding Model):** `jhgan/ko-sbert-nli` (Hugging Face) - 한국어 문장의 의미론적 유사도 측정을 위한 임베딩 생성
*   **벡터 스토어 (Vector Store):** `Chroma DB` - 임베딩된 텍스트 벡터 저장 및 빠른 유사도 검색
*   **리트리버 (Retriever):** `Chroma DB` 기반 Retriever (+ 필요시 Parent Document Retriever, Long Context Reorder 등 고급 전략 적용 고려) - 질문과 관련성 높은 문서 청크 검색
*   **오케스트레이션:** LangChain Expression Language (LCEL) - RAG 체인 구성의 유연성 및 가독성 확보
*   **배포:** Streamlit - 파이썬 기반 웹 애플리케이션 데모 구축 및 배포

**구현 과정 (RAG 파이프라인):**

1.  **문서 로드:** `DocumentLoader`를 사용하여 대한민국 헌법 원문 텍스트 파일을 로드했습니다.
2.  **텍스트 분할:** 로드된 문서를 `RecursiveCharacterTextSplitter`를 이용하여 의미 있는 단위(chunk)로 분할했습니다. `chunk_size`와 `chunk_overlap` 파라미터를 조정하여 검색 성능과 문맥 유지 사이의 균형을 맞추고자 노력했습니다.
3.  **임베딩 생성:** 분할된 텍스트 청크들을 한국어 성능이 우수한 `jhgan/ko-sbert-nli` 모델을 사용하여 벡터로 변환(임베딩)했습니다.
4.  **벡터 저장:** 생성된 임베딩 벡터들을 `Chroma DB`에 저장하여 효율적인 검색이 가능하도록 인덱스를 구축했습니다.
5.  **정보 검색 (Retrieval):** 사용자 질문이 들어오면, 동일한 임베딩 모델을 사용하여 질문을 벡터로 변환한 뒤, `Chroma DB`에 저장된 벡터들과의 유사도(코사인 유사도 등)를 계산하여 가장 관련성이 높은 헌법 텍스트 청크들을 검색(Retrieve)했습니다. (만약 Parent Document Retriever나 Long Context Reorder를 사용했다면, 이 단계에서 어떻게 적용했는지 구체적으로 설명 추가)
6.  **답변 생성 (Generation):** 검색된 관련성 높은 헌법 텍스트 청크(Context)들과 사용자 질문(Query)을 미리 정의된 프롬프트 템플릿에 맞춰 `EEVE-Korean-10.8B` LLM에 전달했습니다. LLM은 주어진 컨텍스트를 기반으로 사용자 질문에 대한 답변을 생성하도록 지시받았습니다.
7.  **체인 구성 (LCEL):** 위 1~6 단계를 LangChain Expression Language (LCEL)을 사용하여 파이프라인으로 매끄럽게 연결하여 코드의 가독성과 유지보수성을 높였습니다.
8.  **웹 애플리케이션 배포:** 구축된 RAG 챗봇 로직을 Streamlit 프레임워크를 사용하여 웹 인터페이스로 구현하고, 사용자가 직접 질문하고 답변을 받을 수 있도록 배포했습니다.

**결과:**

*   **소스 코드 (GitHub):** `https://github.com/timothyseo/langchain_practice`

본 프로젝트를 통해 사용자는 대한민국 헌법에 대한 질문을 자연어로 입력하고, RAG 시스템이 헌법 원문을 참조하여 생성한 신뢰도 높은 답변을 얻을 수 있습니다. 이는 LLM의 창의적인 능력과 정보 검색 기술을 결합하여 특정 지식 기반의 정확한 정보 제공 가능성을 보여줍니다.

**어려웠던 점 및 해결 과정:**

*   초기 검색 정확도가 낮아 `chunk_size`, `chunk_overlap` 파라미터 튜닝 및 다양한 Text Splitter 옵션을 실험하여 개선했습니다.
*   LLM이 주어진 컨텍스트 외의 정보를 바탕으로 답변하는 경향이 있어, 프롬프트 엔지니어링을 통해 컨텍스트 기반 답변 생성을 명확히 지시했습니다.
*   Chroma DB에 데이터 적재 중 메모리 부족으로 인한 커널 충돌 발생하여 Chroma DB 사용 시 데이터를 청크로 나누어 분할 적재했고, 대안으로 FAISS를 사용했습니다.
*   Parent Document Retriever 적용 시, 원본 문서와 청크 간의 연결 및 검색 효율성 확보 방안을 고민했습니다.

**향후 개선 방향:**

*   더 성능이 우수한 한국어 임베딩 모델 또는 LLM 적용 테스트
*   검색 정확도 향상을 위한 HyDE(Hypothetical Document Embeddings) 등 고급 RAG 기법 도입 검토
*   사용자 피드백을 반영한 지속적인 성능 개선
*   답변의 근거가 되는 헌법 조항 명시 기능 강화
