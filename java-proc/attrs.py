l = set()
l = l.union(["package", "imports", "types"])
l = l.union(["path", "static", "wildcard"])
l = l.union(["documentation",])
l = l.union(["modifiers", "annotations"])
l = l.union(["name", "body"])
l = l.union(["name",])
l = l.union(["type_parameters", "extends", "implements"])
l = l.union(["implements",])
l = l.union(["type_parameters", "extends",])
l = l.union(["name", "dimensions",])
l = l.union(["arguments", "sub_type"])
l = l.union(["type", "pattern_type"])
l = l.union(["name", "extends"])
l = l.union(["name", "element"])
l = l.union(["name", "value"])
l = l.union(["values",])
l = l.union(["type_parameters", "return_type", "name", "parameters", "throws", "body"])
l = l.union(["type", "declarators"])
l = l.union(["type_parameters", "name", "parameters", "throws", "body"])
l = l.union(["initializers",])
l = l.union(["type", "declarators"])
l = l.union(["name", "dimensions", "initializer"])
l = l.union(["type", "name", "varargs"])
l = l.union(['name',])
l = l.union(["label",])
l = l.union(["condition", "then_statement", "else_statement"])
l = l.union(["condition", "body"])
l = l.union(["condition", "body"])
l = l.union(["control", "body"])
l = l.union(["condition", "value"])
l = l.union(["goto",])
l = l.union(["goto",])
l = l.union(["expression",])
l = l.union(["expression",])
l = l.union(["lock", "block"])
l = l.union(["resources", "block", "catches", "finally_block"])
l = l.union(["expression", "cases"])
l = l.union(["statements",])
l = l.union(["expression",])
l = l.union(["type", "name", "value"])
l = l.union(["parameter", "block"])
l = l.union(["types", "name"])
l = l.union(["case", "statements"])
l = l.union(["init", "condition", "update"])
l = l.union(["var", "iterable"])
l = l.union(["expressionl", "value", "type"])
l = l.union(["condition", "if_true", "if_false"])
l = l.union(["operator", "operandl", "operandr"])
l = l.union(["type", "expression"])
l = l.union(["expression", "method", "type_arguments"])
l = l.union(['parameters', 'body'])
l = l.union(["prefix_operators", "postfix_operators", "qualifier", "selectors"])
l = l.union(["value",])
l = l.union(["member",])
l = l.union(["type_arguments", "arguments"])
l = l.union(["member",])
l = l.union(["member",])
l = l.union(["member",])
l = l.union(["index",])
l = l.union(["type",])
l = l.union(["type",])
l = l.union(["dimensions", "initializer"])
l = l.union(["constructor_type_arguments", "arguments", "body"])
l = l.union(["constructor_type_arguments", "arguments", "body"])
l = l.union(["constants", "declarations"])
l = l.union(["name", "arguments", "body"])
l = l.union(["name", "return_type", "dimensions", "default"])

for idx,i in enumerate(l):
    print str(i)+" = "+str(idx)