from basecharactor import BaseCharacter


class RabbitCharacter(BaseCharacter):
    def __init__(self, name, age="未知", likes=None, dislikes=None, relationships=None, psychology=None,
                 appearance=None, trait=None, weakness=None, background=None, other=None):
        super().__init__(name, age, likes, dislikes, relationships, psychology, appearance, trait, weakness, background,
                         other)

    def describe(self):

        return super().describe()
