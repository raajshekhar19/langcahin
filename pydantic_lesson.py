from pydantic import BaseModel,EmailStr,Field
from typing import Optional
class Student(BaseModel):
    name: str = 'nitish'
    age:Optional[int] = None
    email:EmailStr
    cgpa:float = Field(gt=0,lt=10,default=5.0)


new_student = {'name':'raaj','age':'32','email':'abc@gmail.com','cgpa':9.0}
student = Student(**new_student)

print(student)