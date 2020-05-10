import logging,os
logger = logging.getLogger(__name__)

class InputExample(object):
    """
    token classification 的输入
    """

    def __init__(self,guid,words,labels):
        self.guid = guid
        self.words = words
        self.labels = labels
    
class InputFeature(object):
    """
    数据特征
    """
    def __init__(self,input_ids,input_mask,segment_ids,label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
    

def read_example_from_file(data_dir,mode):
    file_path = os.path.join(data_dir,"{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path,"r+",encoding="utf-8") as f:
        words = []
        labels = []
        # 一行一行的读取数据
        for line in f:
            if line.startswith("--DOCSTART--") or line == "" or line == '\n':
                if words:
                    examples.append(InputExample(
                        guid = f"{mode}-{guid_index}",
                        words = words,
                        labels = labels
                    ))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(' ')
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace('\n',""))
                else:
                    labels.append("0")
        if words:
            examples.append(InputExample(
                guid = f"{mode}-{guid_index}",
                words = words,
                labels = labels
            ))
    return examples