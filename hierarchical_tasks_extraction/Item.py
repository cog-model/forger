from typing import List


class AbstractItemOrAction(dict):
    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = value

    @property
    def name(self):
        return self.__getitem__('name')

    @name.setter
    def name(self, value):
        self.__setitem__('name', value)

    @property
    def value(self):
        return self.__getitem__('value')

    @value.setter
    def value(self, value):
        self.__setitem__('value', value)

    def is_item(self):
        return self.get('type') == 'item'

    def is_action(self):
        return self.get('type') == 'action'


class Action(AbstractItemOrAction):

    def __init__(self, name, value):
        super().__init__(name, value)
        self.__setitem__('type', 'action')

    def is_noop(self):
        return not self.value


class Item(AbstractItemOrAction):

    def __init__(self, name: str, value: int, begin: int, end: int, actions: List[Action] = ()):
        super().__init__(name, value)
        self.actions = actions
        self.begin = begin
        self.end = end
        self.__setitem__('type', 'item')

    @property
    def actions(self) -> List[Action]:
        return self.__getitem__('actions')

    @actions.setter
    def actions(self, value: List[Action]):
        self.__setitem__('actions', value)

    def get_last_action(self) -> Action:
        actions = self.__getitem__('actions')
        if actions:
            return actions[-1]
        else:
            return None

    def add_action(self, action: Action):
        self.actions = (*self.actions, action)

    @property
    def begin(self):
        return self.__getitem__('begin')

    @begin.setter
    def begin(self, value):
        self.__setitem__('begin', value)

    @property
    def end(self):
        return self.__getitem__('end')

    @end.setter
    def end(self, value):
        self.__setitem__('end', value)