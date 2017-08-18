#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This is for preprocess AST from Java Corpus """
from __future__ import print_function
import sys
import traceback
import javalang

reload(sys)
sys.setdefaultencoding('utf-8')
print(javalang.__path__)

def render_ast_innate(ast):
    return ast.to_java()

def print_ast(ast):
    """ Print the ast in order to debug"""

    # TODO
    def get_type_str(para):
        # print(para.type.children)
        if hasattr(para, 'type'):
            name = para.type.name;
            dimensions = para.type.dimensions
            name += "[]"*len(dimensions)
            return name
        else:
            return ""

    def get_modifiers_str(node):
        if hasattr(node, "modifiers"):
            modifiers = getattr(node, "modifiers") 
            modifiers = " ".join(modifiers)
            return modifiers
        else:
            return ""

    # TODO: implement varargs
    def get_parameter_str(node):
        if hasattr(node, 'parameters'):
            parameters = getattr(node, "parameters")
        else:
            return ""

        paras = []
        for para in parameters:
            # print(para.type.children)
            name_str = getattr(para, 'name')
            type_str = get_type_str(para)
            modifiers_str =  get_modifiers_str(para)
            varargs = getattr(para, 'varargs')
            paras.append(modifiers_str+' '+type_str+' '+name_str)

        return ", ".join(paras)

    def get_declarators_str(node):
        if hasattr(node, 'declarators'):
            d_list = []
            for declarator in getattr(node, "declarators"):
                name = declarator.name;
                dimensions = declarator.dimensions;
                name += "[]"*len(dimensions)

                # TODO: deal with initializer
                initializer = "new XX()"
                d_list.append(name+' = '+initializer)

            return ", ".join(d_list)
        else:
            return ""

    stack = [ast]
    while stack:
        node = stack.pop()
        #print(node)
        # TODO: add documentation expansion
        # PackageDeclaration
        if isinstance(node, javalang.tree.PackageDeclaration):
            print('package', getattr(node, 'name')+';')
            continue
        # Import
        # TODO: Add static and wildcard representation
        if isinstance(node, javalang.tree.Import):
            print('import', getattr(node, 'path')+';')
            continue
        # TODO: type_parameters
        if isinstance(node, javalang.tree.ClassDeclaration):
            name = getattr(node, 'name')
            extends = getattr(node, 'extends')
            implements = getattr(node, 'implements')
            type_parameters = getattr(node, 'type_parameters')
            body = getattr(node, 'body')
            modifiers_str = get_modifiers_str(node)
            adding_ext = ""
            adding_imp = ""
            if extends:
                adding_ext = 'extends '+getattr(extends, 'name')
            if implements:
                adding_imp = 'implements '+", ".join(map(lambda d: getattr(d, 'name'), implements))
            print(modifiers_str, 'class', name, adding_ext, adding_imp)
            stack.extend(['}', body, '{'])
            continue

        if isinstance(node, javalang.tree.MethodDeclaration):
            modifiers_str = get_modifiers_str(node)
            return_type = getattr(node, "return_type")

            # TODO: deal more with return type
            return_type_str = return_type.name
            name_str = getattr(node, 'name')
            parameters_str = get_parameter_str(node) 
            body = getattr(node, "body")

            print(modifiers_str, return_type_str, name_str,'('+parameters_str+')')

            stack.extend(['}', body, '{'])
            continue

        if isinstance(node, javalang.tree.ConstructorDeclaration):
            modifiers_str = get_modifiers_str(node)
            name_str = getattr(node, 'name')
            parameters_str = get_parameter_str(node) 
            body = getattr(node, "body")

            print(modifiers_str, name_str,'('+parameters_str+')')

            stack.extend(['}', body, '{'])
            continue

        if isinstance(node, javalang.tree.FieldDeclaration):
            modifiers_str = get_modifiers_str(node)
            # TODO: Deal more with type
            type_str = getattr(node, 'type').name
            declarators_str_list = get_declarators_str(node)
            modifiers_str = get_modifiers_str(node)

            print(modifiers_str, type_str, declarators_str_list+';')
            continue

        if isinstance(node, javalang.tree.FormalParameter):
            modifiers_str = get_modifiers_str(node)

        if isinstance(node, str):
            print(node)
        if isinstance(node, list):
            stack.extend(node[::-1])
        elif hasattr(node, 'children'):
            stack.extend(node.children[::-1])

def check_file(line):
    """ Give a check on file, whether it can be parsed or not """
    try:
        javalang.parse.parse(line)
        return {'Status':True, 'Msg':None}
    except javalang.parser.JavaSyntaxError as error:
        trace_back = traceback.format_exc()
        print(trace_back)
        return {'Status':False, 'Msg':error.message}

def preprocess(path):
    """ Preprocess all lines in one file"""
    with open(path, 'r') as test_file:
        lines = test_file.readlines()

    for i, line in enumerate(lines):
        if i < 5960:
            continue
        result = check_file(line)
        if not result['Status']:
            print('Line %d failed with ERROR: %s'%(i, result['Msg']))
            return False
        else:
            print('Line %d parsing good'%i)

if __name__ == "__main__":
    # preprocess('./data/java_1M_train')

    with open('./test.java','r') as input_file:
        program = input_file.read()

    ast = javalang.parse.parse(program)

    result1 = render_ast_innate(ast) 
    print(result1)

    ast1 = javalang.parse.parse(result1)
    result2 = render_ast_innate(ast1)
    ast2 = javalang.parse.parse(result2)

    if result1 == result2: 
        print("SUCCESS")
    else:
        print("FAIL")