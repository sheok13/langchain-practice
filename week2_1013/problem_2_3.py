from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class StudentInfo(BaseModel):
    name: str = Field(description="학생의 이름")
    age: int = Field(description="학생의 나이")
    major: str = Field(description="학생의 전공")
    hobbies: List[str] = Field(description="학생의 취미 리스트")
    goal: str = Field(description="학생의 목표")

def solve_2_3(text):
    parser = PydanticOutputParser(pydantic_object=StudentInfo)
    prompt = PromptTemplate(
        template="다음 텍스트에서 학생 정보를 추출해주세요.\n텍스트: {text}\n{format_instructions}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    
    print("--- [문제 2-3 실행 결과] ---")
    result = chain.invoke({"text": text})
    print(result.json(indent=4, ensure_ascii=False))

if __name__ == "__main__":
    solve_2_3("안녕하세요! 저는 김민수이고 22살입니다. 컴퓨터공학을 전공하고 있어요. 취미로는 게임하기, 영화보기, 코딩을 좋아합니다. 앞으로 훌륭한 개발자가 되는 것이 목표입니다.")