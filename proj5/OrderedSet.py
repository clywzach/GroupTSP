import collections

class OrderedSet(collections.Set):

    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __iter__(self):
        return iter(self.d)

    def __contains__(self, el):
        return el in self.d

    def __len__(self):
        return len(self.d)
