# 判断一个类是否拥有构造函数
"""
class a:
    def __init__(self):
        self.aa = "sdfdsf"
        
class b:
    bb = "bb"


False : a.__init__ == object.__init__ 
True  : b.__init__ == object.__init__
"""

# 检测一个类的构造函数的参数
"""
signature = inspect.signature(cls.__init__)
parameters = signature.parameters
"""

# 懒加载
"""
import datetime
import time
def lazy_read():
    result = iter(read("./cls.py"))

    print(result.__next__())

    print("----------")
    # for i in result:
    #     time.sleep(1)
    #     a = i
    
    result = list(read("./cls.py"))
    # for i in result:
    #     # time.sleep(1)
    #     a = i


def read(file_path):
    with open(file_path,"r+",encoding="utf-8") as f:
        for line in f:
            print(datetime.datetime.now())
            yield line

lazy_read()
"""

# 使用defaultdict 构建树形字典结构
def build_tree_dict():
    from collections import defaultdict
    import json

    tree = lambda: defaultdict(tree)

    tree_dict = tree()
    tree_dict["111"]["222"]["333"] = "444"
    
    print(json.dumps(tree_dict))
