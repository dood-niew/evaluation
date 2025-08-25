import re

class PostProcessor:
    def __init__(self, text):
        self.text = text
    
    def extract_think(self):
        think_match = re.search(r'<think>(.*?)</think>', self.text, re.DOTALL)
        return think_match.group(1).strip() if think_match else None
    
    def extract_answer(self):
        answer_match = re.search(r'<answer>(.*?)</answer>', self.text, re.DOTALL)
        return answer_match.group(1).strip() if answer_match else None
    
    def extract_boxed_number(self):
        boxed_match = re.search(r'\\?boxed\{([^}]+)\}', self.extract_answer() or '')
        return boxed_match.group(1) if boxed_match else None

