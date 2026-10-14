from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import EnumOutputParser
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Emotion(Enum):
    POSITIVE = "긍정"
    NEGATIVE = "부정"
    NEUTRAL = "보통"

def solve_2_2(review):
    parser = EnumOutputParser(enum=Emotion)
    prompt = PromptTemplate.from_template(
        "다음 영화 리뷰의 감정을 분석해주세요.\n리뷰: {review}\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser
    
    print(f"--- [문제 2-2 실행 결과] ---")
    result = chain.invoke({"review": review})
    print(f"리뷰: {review}\n감정: {result.value}")

if __name__ == "__main__":
    solve_2_2("배우들의 연기가 훌륭하고 스토리도 감동적이었어요!")