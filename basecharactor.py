class BaseCharacter:

    def __init__(self, name, age="未知", likes=None, dislikes=None,relationships=None,psychology=None,appearance=None,
                 trait=None,weakness=None,background=None,other=None):
        self.name = name  # 角色名字
        self.age = age  # 角色年龄
        self.likes = likes  # 角色喜好
        self.dislikes = dislikes  # 角色厌恶的事物
        self.relationships = relationships  # 关系
        self.psychology = psychology  # 心理
        self.appearance = appearance  # 外貌
        self.trait = trait  # 特质
        self.weakness = weakness  # 弱点
        self.background = background  # 背景
        self.other = other  # 其他


    def describe(self):
        description = (f"name:{self.name},age:{self.age},like:{self.likes},dislike:{self.dislikes},relationships:{self.relationships},"
                       f"psychology:{self.psychology},appearance:{self.appearance},trait:{self.trait},weakness:{self.weakness},background:{self.background},other:{self.other}")
        return description
