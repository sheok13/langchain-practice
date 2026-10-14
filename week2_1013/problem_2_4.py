from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv
import json

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def solve_2_4(text):
    response_schemas = [
        ResponseSchema(name="destination", description="여행지"),
        ResponseSchema(name="duration", description="여행 기간"),
        ResponseSchema(name="budget", description="총 예산"),
        ResponseSchema(name="rating", description="추천도 (1-5점)"),
        ResponseSchema(name="activities", description="주요 활동 리스트", type="list")
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    prompt = PromptTemplate(
        template="다음 여행 후기에서 정보를 추출해주세요.\n후기: {text}\n{format_instructions}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    
    print("--- [문제 2-4 실행 결과] ---")
    result = chain.invoke({"text": text})
    print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    solve_2_4("지난 주에 부산으로 2박 3일 여행을 다녀왔어요. 총 30만원 정도 썼는데 해운대에서 바다구경하고, 자갈치시장에서 회 먹고, 감천문화마을도 구경했어요. 정말 만족스러운 여행이었습니다. 5점 만점에 4점 정도 줄 수 있을 것 같아요.")