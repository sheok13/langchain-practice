from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def solve_1_3(news_text):
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{news}"),
        ("ai", "키워드: {keywords}")
    ])
    
    examples = [
        {
            "news": "삼성전자가 내년 초에 자체적으로 개발한 인공지능(AI) 가속기를 처음으로 출시할 예정이다. 이는 AI 반도체 시장에서 지배적인 위치를 차지하고 있는 엔비디아의 독점을 도전하고, 세계 최고의 반도체 제조업체로서의 지위를 다시 확립하려는 삼성전자의 노력으로 해석된다.",
            "keywords": "삼성전자, 인공지능, 엔비디아"
        },
        {
            "news": "세계보건기구(WHO)는 최근 새로운 건강 위기에 대응하기 위해 국제 협력의 중요성을 강조했다. 전염병 대응 역량의 강화와 글로벌 보건 시스템의 개선이 필요하다고 발표했다.",
            "keywords": "세계보건기구, 건강위기, 국제협력"
        }
    ]
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "뉴스 키워드 추출 전문가입니다. 핵심 키워드 3개를 추출하세요."),
        few_shot_prompt,
        ("human", "{input}")
    ])
    
    chain = final_prompt | llm | StrOutputParser()
    
    print("--- [문제 1-3 실행 결과] ---")
    print(chain.invoke({"input": news_text}))

if __name__ == "__main__":
    solve_1_3("제미나이 2.0 플래시는 현재 구글 AI 스튜디오(Google AI Studio) 및 버텍스 AI(Vertex AI)에서 제미나이 API를 통해 개발자에게 실험 모델로 제공됩니다. 모든 개발자는 멀티모달 입력 및 텍스트 출력을 사용할 수 있으며, 텍스트 음성 변환(text-to-speech) 및 네이티브 이미지 생성은 일부 파트너들을 대상으로 제공됩니다. 내년 1월에는 더 많은 모델 사이즈와 함께 일반에 공개될 예정입니다.")
