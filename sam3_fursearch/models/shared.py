import inspect
from typing import ClassVar


class SharedInstance:
    """Mixin that caches instances by constructor arguments.

    Identical arguments return the same object; __init__ is skipped on
    subsequent calls. Each subclass gets its own cache via __init_subclass__.

    Subclasses must add this guard at the top of __init__:
        if self._initialized:
            return
        self._initialized = True
    """

    _instances: ClassVar[dict] = {}
    _initialized: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._instances = {}

    def __new__(cls, *args, **kwargs):
        key = cls._make_key(args, kwargs)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]

    @classmethod
    def _make_key(cls, args, kwargs):
        sig = inspect.signature(cls.__init__)
        bound = sig.bind(None, *args, **kwargs)  # None stands in for self
        bound.apply_defaults()
        params = list(bound.arguments.items())[1:]  # drop self
        key_parts = []
        for k, v in params:
            try:
                hash(v)
                key_parts.append((k, v))
            except TypeError:
                key_parts.append((k, id(v)))
        return tuple(key_parts)
