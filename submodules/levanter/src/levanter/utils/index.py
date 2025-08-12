from typing import Generic, Iterable, Iterator, TypeVar


T = TypeVar("T")


class Index(Generic[T]):
    """
    Index is a bidirectional mapping from (incremental) integers to objects.

    Needs to be fast, so it exposes the underlying data structures.
    """

    def __init__(self, objs: Iterable[T] = ()):
        self._index_to_obj: list[T] = []
        self._obj_to_index: dict[T, int] = {}
        for obj in objs:
            self.append(obj)

    def __len__(self):
        return len(self._index_to_obj)

    def __getitem__(self, index: int) -> T:
        return self._index_to_obj[index]

    def __setitem__(self, index: int, obj: T):
        self._index_to_obj[index] = obj
        self._obj_to_index[obj] = index

    def append(self, obj: T) -> int:
        index = len(self)
        self._index_to_obj.append(obj)
        self._obj_to_index[obj] = index
        return index

    def get_index(self, obj: T) -> int:
        return self._obj_to_index[obj]

    def get_obj(self, index: int) -> T:
        return self._index_to_obj[index]

    def __contains__(self, obj: T) -> bool:
        return obj in self._obj_to_index

    def __iter__(self) -> Iterator[T]:
        return iter(self._index_to_obj)
