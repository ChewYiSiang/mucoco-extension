def generic_function1(var1):
    res, switch = ([], True)
    while var1:
        res.append(min(var1) if switch else max(var1))
        var1.remove(res[-1])
        switch = not switch
    return res
print(generic_function1([1, 2, 3, 4]))