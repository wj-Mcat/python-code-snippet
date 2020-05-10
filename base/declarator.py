"""
使用类来作为装饰器，提升其扩展性
"""
from functools import wraps
from overrides import overrides

class Log:

    def __init__(self,log_file = "./log/info.log"):
        self.log_file = log_file

    def __call__(self,func):
        @wraps(func)
        def inner_func(*args,**kwargs):
            print("before func ...")
            func(*args,**kwargs)
            print("after func ...")
            self.notify(*args,**kwargs)
    
        return inner_func
    
    def notify(self,*args,**kwargs):
        pass



class EmailLog(Log):
    @overrides
    def notify(self,*args,**kwargs):
        print("send email to developer ...")


from typing import Type, TypeVar
T = TypeVar("T", bound = "A")
class A:
    @classmethod
    def say(cls, contact_id: str):
        if isinstance(cls, A):
            print("1111")
        if issubclass(cls, A):
            print("222")
        print(cls.__class__)

class B(A):

    @classmethod
    @overrides
    def say(cls, contact_id: str):
        if isinstance(cls, A):
            print("1111")
        if issubclass(cls,A):
            print("222")
        print(cls.__class__)

A.say("123123")
a = A()
a.say("qwertyuio")


B.say("123456789")
b = B()
b.say("zxcvbnm,.")