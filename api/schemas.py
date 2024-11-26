from fastapi import Query
from pydantic import BaseModel

class Item(BaseModel):
    text: str

class TTSParams():
    def __init__(
        self,
        # filter: str = Query('4',
        #                                description="Фильтр",
        #                                example="4"),
        lang: str = Query('ru',
                          description="Язык:ru,en,it,fr,ja,zh-cn,kaz,grc",
                          example="ru"),
        ssml: bool = Query(default=False,
                           description="Если был передан ssml файл на английском языке, то установите True",
                           ),
    ):
        # self.filter = filter
        self.lang = lang
        self.ssml = ssml
