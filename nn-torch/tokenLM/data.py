import os
import javalang
import torch
import codecs

class ASTProvider(object):
    """
        at first, set batch size = 1
        By asking a current node position, generate next token

        Situation 1:
            
    """
    def __init__(self, path):
        if not os.path.exists(path):
            raise EnvironmentError("Path not exist")
        
    def iterator_ast():
