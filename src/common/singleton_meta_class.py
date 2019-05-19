class Singleton(type):
    """
    Singleton metaclass
    Based on Python Cookbook 3rd Edition Recipe 9.13
    Only one instance of a class can exist. Does not work with __slots__
    """

    def __init__(cls, *args, **kw):
        super(Singleton, cls).__init__(*args, **kw)
        cls.__instance = None

    def __call__(cls, *args, **kw):
        if cls.__instance is None:
            cls.__instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.__instance
