from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def solve_2_1(topic):
    parser = CommaSeparatedListOutputParser()
    prompt = PromptTemplate(
        template="{topic}와(과) 관련된 한국의 유명한 장소나 활동 5가지를 추천해주세요.\n{format_instructions}",
        input_variables=["topic"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    print("--- [문제 2-1 실행 결과] ---")
    result = chain.invoke({"topic": topic})
    print(result)

if __name__ == "__main__":
    solve_2_1("음식")