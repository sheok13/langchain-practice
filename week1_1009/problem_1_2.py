from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def solve_1_2(city):
    prompt1 = ChatPromptTemplate.from_messages([
        ("system", "당신은 훌륭한 여행 가이드입니다."),
        ("user", "{city}의 대표적인 관광 명소 1곳의 이름만 추천해주세요.")
    ])
    chain1 = prompt1 | llm | StrOutputParser()
    
    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "당신은 훌륭한 여행 가이드입니다."),
        ("user", "{attraction}에 대한 역사, 특징, 방문 팁을 상세히 알려주세요.")
    ])
    
    multi_chain = (
        {"attraction": chain1} 
        | RunnablePassthrough.assign(details=(prompt2 | llm | StrOutputParser()))
    )
    
    print("--- [문제 1-2 실행 결과] ---")
    result = multi_chain.invoke({"city": city})
    print(f"1단계 결과 (추천 명소): {result['attraction']}\n")
    print(f"2단계 결과 (상세 정보):\n{result['details']}")

if __name__ == "__main__":
    solve_1_2("로마")
