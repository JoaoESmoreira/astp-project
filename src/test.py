def myFun(**kwargs):
    for key, value in kwargs.items():
        print("%s == %s" % (key, value))
    if first in kwargs:
        print()


# Driver code
myFun(first='Geeks', mid='for', last='Geeks')