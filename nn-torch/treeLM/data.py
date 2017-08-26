from __future__ import print_function
from enum import Enum, unique
import os
import traceback
import argparse
import codecs
import javalang
import torch

@unique
class AttrMapping(Enum):
    NONE = -1
    pattern_type = 0
    statements = 1
    expressionl = 2
    operator = 3
    iterable = 4
    parameters = 5
    init = 6
    extends = 7
    constants = 8
    imports = 9
    var = 10
    type = 11
    annotations = 12
    resources = 13
    qualifier = 14
    constructor_type_arguments = 15
    cases = 16
    condition = 17
    else_statement = 18
    case = 19
    catches = 20
    modifiers = 21
    name = 22
    default = 23
    declarations = 24
    value = 25
    prefix_operators = 26
    values = 27
    finally_block = 28
    control = 29
    lock = 30
    type_arguments = 31
    initializer = 32
    static = 33
    element = 34
    then_statement = 35
    dimensions = 36
    label = 37
    member = 38
    arguments = 39
    parameter = 40
    method = 41
    body = 42
    package = 43
    goto = 44
    index = 45
    type_parameters = 46
    update = 47
    if_true = 48
    declarators = 49
    varargs = 50
    path = 51
    return_type = 52
    sub_type = 53
    types = 54
    implements = 55
    if_false = 56
    postfix_operators = 57
    operandr = 58
    documentation = 59
    selectors = 60
    initializers = 61
    wildcard = 62
    operandl = 63
    expression = 64
    throws = 65
    block = 66

    @staticmethod
    def size(self):
        return 68

@unique
class NodeMapping(Enum):
    UNK = 0
    LIST_END = 1
    STRING = 2
    NONE = 3
    CompilationUnit = 10
    Import = 11
    Documented = 12
    Declaration = 13
    TypeDeclaration = 14
    PackageDeclaration = 15
    ClassDeclaration = 16
    EnumDeclaration = 17
    InterfaceDeclaration = 18
    AnnotationDeclaration = 19
    Type = 20
    BasicType = 21
    ReferenceType = 22
    TypeArgument = 23
    TypeParameter = 24
    Annotation = 25
    ElementValuePair = 26
    ElementArrayValue = 27
    Member = 28
    MethodDeclaration = 29
    FieldDeclaration = 30
    ConstructorDeclaration = 31
    ConstantDeclaration = 32
    ArrayInitializer = 33
    VariableDeclaration = 34
    LocalVariableDeclaration = 35
    VariableDeclarator = 36
    FormalParameter = 37
    InferredFormalParameter = 38
    Statement = 39
    IfStatement = 40
    WhileStatement = 41
    DoStatement = 42
    ForStatement = 43
    AssertStatement = 44
    BreakStatement = 45
    ContinueStatement = 46
    ReturnStatement = 47
    ThrowStatement = 48
    SynchronizedStatement = 49
    TryStatement = 50
    SwitchStatement = 51
    BlockStatement = 52
    StatementExpression = 53
    TryResource = 54
    CatchClause = 55
    CatchClauseParameter = 56
    SwitchStatementCase = 57
    ForControl = 58
    EnhancedForControl = 59
    Expression = 60
    Assignment = 61
    TernaryExpression = 62
    BinaryOperation = 63
    Cast = 64
    MethodReference = 65
    LambdaExpression = 66
    Primary = 67
    Literal = 68
    This = 69
    MemberReference = 70
    Invocation = 71
    ExplicitConstructorInvocation = 72
    SuperConstructorInvocation = 73
    MethodInvocation = 74
    SuperMethodInvocation = 75
    SuperMemberReference = 76
    ArraySelector = 77
    ClassReference = 78
    VoidClassReference = 79
    Creator = 80
    ArrayCreator = 81
    ClassCreator = 82
    InnerClassCreator = 83
    EnumBody = 84
    EnumConstantDeclaration = 85
    AnnotationMethod = 86

    @staticmethod
    def size(self):
        return 87

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
    LIST_END = "LIST_END"
    SUBT_STR = "<SUBT_STR>"
    SUBT_END = "<SUBT_END>"
    STRING = "STRING"
    NONE = "NONE"
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
                    yield node
            except:
                print(traceback.format_exc())
                continue
        
    def iterate_ast(self, node, attr="NONE"):
        if isinstance(node, type(None)):
            yield {"attr": attr, "node": self.NONE}

        if isinstance(node, list):
            # yield self.LIST_STR
            for e in node:
                for i in self.iterate_ast(e, attr):
                    yield i
            yield {"attr": attr, "node": self.LIST_END}

        if isinstance(node, javalang.ast.Node):
            yield {"attr": attr, "node": node}
            # yield self.SUBT_STR
            for idx, child in enumerate(node.children):
                for i in self.iterate_ast(child, node.attrs[idx]):
                    yield i
            # yield self.SUBT_END

        if isinstance(node, str) or (isinstance(node, unicode)):
            yield {"attr":attr, "node": self.STRING}
        
    def print_node(self, node):
        print(AttrMapping[node["attr"]].value, NodeMapping[str(node["node"])].value) 
        # pass

class Saver(object):
    """
        Provide an output interface
    """
    def __init__(self, logpath):
        if not (os.path.exists(logpath) and os.path.isdir(logpath)):
            raise EnvironmentError("Log Path Not Find")
        self.logpath = logpath

    def save(self, model):
        cur_time = time.strftime("%B%d-%H%M%S")
        filepath = os.path.join(self.logpath, cur_time+".pth")
        torch.save(model, filepath)

        print("Save as file: {}".format(filepath))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--filename", type=str, default="/Users/zyxu/working/Tree-LSTM-LM/nn-torch/treeLM/test.java")
    argparser.add_argument("--cuda", action="store_true")
    args = argparser.parse_args()

    provider = JavaASTProvider(args.filename)

    provider.iterate()
