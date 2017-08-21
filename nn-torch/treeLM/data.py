from __future__ import print_function
import os
import argparse
import javalang
import torch
import codecs

class JavaASTProvider(object):
    """
        at first, set batch size = 1
        By asking a current node position, generate next token

        Situation 1:
            Attr -> Node
        Situation 2:
            Attr -> Terminal
        Situation 3:
            Attr -> List
    """
    LIST_STR = "<LIST_STR>"
    LIST_END = "<LIST_END>"
    SUBT_STR = "<SUBT_STR>"
    SUBT_END = "<SUBT_END>"
    STRING = "<STR>"
    NONE_STR = "<NONE>"
    def __init__(self, filename):

        if not os.path.exists(filename):
            raise EnvironmentError("File does not exist")
        
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            self.java_files = [f.read()]

    def iterate(self):
        for java_file in self.java_files:
            try:
                ast = javalang.parse.parse(java_file)
                # print(ast)
                for node in self.iterate_ast(ast):
                    self.print_node(node)
            except:
                continue
        
    def iterate_ast(self, node):
        if isinstance(node, type(None)):
            yield self.NONE_STR

        if isinstance(node, list):
            yield self.LIST_STR
            for e in node:
                for i in self.iterate_ast(e):
                    yield i
            yield self.LIST_END

        if isinstance(node, javalang.ast.Node):
            yield node
            # yield self.SUBT_STR
            for child in node.children:
                for i in self.iterate_ast(child):
                    yield i
            yield self.SUBT_END

        if isinstance(node, str) or (isinstance(node, unicode)):
            yield self.STRING
        

    def print_node(self, node):
        print(node)
        # pass


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--filename", type=str, default="/Users/zyxu/working/Tree-LSTM-LM/nn-torch/treeLM/test.java")
    argparser.add_argument("--cuda", action="store_true")
    args = argparser.parse_args()

    provider = JavaASTProvider(args.filename)

    provider.iterate()
