from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 문서 포맷팅 헬퍼 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag_system():
    # 0단계: 문서 로드 (경로 주의: 데이터 폴더 위치 확인)
    loader = PyPDFLoader('../data/콘텐츠분쟁해결_사례.pdf')
    docs = loader.load()

    # 1단계: 문서 분할 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=[
            "\n【사건개요】",
            "\n【쟁점사항】",
            "\n【처리경위】",
            "\n【처리결과】",
            "\n■", "\n\n", "\n", ".", " ", ""
        ]
    )
    split_docs = text_splitter.split_documents(docs)

    # 2단계: 임베딩 모델 설정
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

    # 벡터스토어 생성 (FAISS 활용)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 3단계: 검색기 설정
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # 4단계: LLM 설정
    llm = ChatUpstage(
        model="solar-pro3",
        base_url="https://api.upstage.ai/v1",
        temperature=0.2
    )

    # 5단계: 법률 자문 프롬프트 작성
    prompt_template = PromptTemplate.from_template(
        """당신은 콘텐츠 분야 전문 법률 자문사입니다.
아래 분쟁조정 사례들을 바탕으로 정확하고 전문적인 법률 조언을 제공해주세요.

관련 분쟁사례:
{context}

상담 내용: {question}

답변 가이드라인:
1. 제시된 사례들을 근거로 답변하세요.
2. 관련 법령이나 조항이 있다면 명시하세요.
3. 비슷한 사례의 처리경위와 결과를 참고하여 설명하세요.
4. 실무적 해결방안을 단계별로 제시하세요.
5. 사례에 없는 내용은 "제시된 사례집에서는 확인할 수 없습니다"라고 명시하세요.

전문 법률 조언:"""
    )

    # 6단계: LCEL 체인 생성
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # 7단계: 테스트 질문 실행 예시
    test_questions = [
        "온라인 게임에서 시스템 오류로 아이템이 사라졌는데, 게임회사가 복구를 거부하고 있습니다. 어떻게 해결할 수 있나요?",
        "인터넷 강의를 중도 해지하려고 하는데 과도한 위약금을 요구받고 있습니다. 정당한가요?"
    ]

    print("--- [콘텐츠 분쟁해결 RAG 시스템 테스트] ---")
    for q in test_questions:
        print(f"\n[질문]: {q}")
        response = qa_chain.invoke(q)
        print(f"[답변]:\n{response}\n" + "-"*50)

# 8단계: 분쟁 유형 분류 함수
def classify_dispute_type(query):
    game_keywords = ["게임", "아이템", "계정", "캐릭터", "레벨", "길드", "온라인게임"]
    elearning_keywords = ["강의", "온라인교육", "이러닝", "수강", "환불", "화상교육"]
    web_keywords = ["웹사이트", "무료체험", "자동결제", "구독", "사이트"]
    
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in game_keywords):
        return "게임"
    elif any(keyword in query_lower for keyword in elearning_keywords):
        return "이러닝"
    elif any(keyword in query_lower for keyword in web_keywords):
        return "웹콘텐츠"
    else:
        return "기타"

if __name__ == "__main__":
    # 실행 테스트 (데이터 경로 및 API 키가 준비된 경우 주석 해제하여 실행)
    # run_rag_system()
    pass