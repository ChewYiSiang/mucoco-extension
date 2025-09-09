def multiprocessing_method(func):
    func._tag = "multiprocessing method"
    return func