from allennlp.common.params import Params

# 从文件中创建Param类
"""
params = Params.from_file(params_file=args.config_file_path)
"""

# 从文件中获取不同种类的task
"""
task_keys = [key for key in params.keys() if re.search("^task_", key)]
"""