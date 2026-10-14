from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def solve_1_1(ingredients):
    prompt = PromptTemplate.from_template(
        "{ingredients}로 만들 수 있는 요리를 추천하고 간단한 레시피를 알려주세요."
    )
    chain = prompt | llm | StrOutputParser()
    print("--- [문제 1-1 실행 결과] ---")
    print(chain.invoke({"ingredients": ingredients}))

if __name__ == "__main__":
    solve_1_1("계란, 밥, 김치")
