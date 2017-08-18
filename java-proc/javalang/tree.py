# -*- coding: utf-8 -*-
from .ast import Node

# ------------------------------------------------------------------------------
SEPERATOR = "\r\n"

class CompilationUnit(Node):
    """
        package: Node - PackageDeclaration
        imports: Node - Import
        types:  List of Node
    """
    attrs = ("package", "imports", "types")
    
    def to_java(self):
        package_str = self.package.to_java()
        imports = SEPERATOR.join([i.to_java() for i in self.imports])
        types = SEPERATOR.join([i.to_java() for i in self.types])
        value = package_str + SEPERATOR*2 + imports + SEPERATOR*2 + types

        return value

class Import(Node):
    """
        path: String of import path
        static: Boolean of whether add static
        wildcard: Boolean of whether add ".*"
    """
    attrs = ("path", "static", "wildcard")

    def to_java(self):
        value = 'import '
        if self.static:
            value = value + 'static '
        value = value + self.path
        if self.wildcard:
            value = value + '.*'
        value = value + ';'
        return value

class Documented(Node):
    """
        documentation: String of documentation
    """
    attrs = ("documentation",)

class Declaration(Node):
    """
        modifers: List of String (of modifiers)
        annotations: List of Node - AnnotationDeclaration
    """
    attrs = ("modifiers", "annotations")
    
    # TODO: consider order or the modifiers
    @property
    def modifiers_str(self):
        if self.modifiers:
            return " ".join(self.modifiers)+" "
        else:
            return ""

    @property
    def annotations_str(self):
        if self.annotations:
            annotations = [i.to_java() for i in self.annotations]
            value = SEPERATOR.join(annotations)+SEPERATOR     
        else:
            value = ""
        return value

class TypeDeclaration(Declaration, Documented):
    """
        name: String of type name
        body: List of Node
    """
    attrs = ("name", "body")

    @property
    def fields(self):
        return [decl for decl in self.body if isinstance(decl, FieldDeclaration)]

    @property
    def methods(self):
        return [decl for decl in self.body if isinstance(decl, MethodDeclaration)]

    @property
    def constructors(self):
        return [decl for decl in self.body if isinstance(decl, ConstructorDeclaration)]

class PackageDeclaration(Declaration, Documented):
    """
        name: String of the package name
    """
    attrs = ("name",)

    def to_java(self):
        name = self.name
        value = self.modifiers_str + 'package '+ name +';'
        return value

class ClassDeclaration(TypeDeclaration):
    """
        type_parameters:
        extends: None or Type(ReferenceType)
        implements: None or List of Type(ReferenceType) 
    """
    attrs = ("type_parameters", "extends", "implements")

    def to_java(self):
        name = self.name 
        extends = self.extends
        implements = self.implements 
        # TODO:Implement this
        type_parameters = self.type_parameters
        body = [node.to_java() for node in self.body]
        adding_ext = ""
        if extends:
            adding_ext = " extends "+ extends.to_java()

        adding_imp = ""
        if implements:
            adding_imp = " implements " + ", ".join([t.to_java() for t in implements])

        value = self.annotations_str + self.modifiers_str + "class " +name+adding_ext+adding_imp
        value += SEPERATOR+"{"+SEPERATOR+SEPERATOR.join(body)+SEPERATOR+"}"
        return value

class EnumDeclaration(TypeDeclaration):
    attrs = ("implements",)

class InterfaceDeclaration(TypeDeclaration):
    attrs = ("type_parameters", "extends",)

class AnnotationDeclaration(TypeDeclaration):
    attrs = ()

# ------------------------------------------------------------------------------

class Type(Node):
    """
        name: String of name
        dimensions: List of None or Node(Liberals or else)
    """
    attrs = ("name", "dimensions",)

    @property
    def name_and_dimensions_str(self):
        return get_name_and_dimensions_str(self)

    def to_java(self):
        return self.name_and_dimensions_str

class BasicType(Type):
    """
        simple implementation of type
    """
    attrs = ()

    def to_java(self):
        return self.name_and_dimensions_str

class ReferenceType(Type):
    """
        arguments: ?
        sub_type: ?
    """
    attrs = ("arguments", "sub_type")

    def to_java(self):
        #TODO: Implement this
        arguments = self.arguments
        sub_type = self.sub_type

        return self.name_and_dimensions_str 

class TypeArgument(Node):
    attrs = ("type", "pattern_type")

# ------------------------------------------------------------------------------

class TypeParameter(Node):
    attrs = ("name", "extends")

# ------------------------------------------------------------------------------

class Annotation(Node):
    """
        name: String of name
        element: None or ?
    """
    attrs = ("name", "element")
    #TODO: element
    def to_java(self):
        value = '@'+self.name
        return value

class ElementValuePair(Node):
    attrs = ("name", "value")

class ElementArrayValue(Node):
    attrs = ("values",)

# ------------------------------------------------------------------------------

class Member(Documented):
    attrs = ()

class MethodDeclaration(Member, Declaration):
    """
        type_parameters:?
        return_type: Node - Type(ReferenceType)
        name: String of name
        parameters: List of parameters(FormalParameter)
        throws:?
        body: List of Nodes
    """
    attrs = ("type_parameters", "return_type", "name", "parameters", "throws", "body")

    @property
    def parameters_str(self):
        return get_parameter_str(self)

    def to_java(self):
        name_str = self.name
        parameters_str = self.parameters_str
        modifiers_str = self.modifiers_str
        return_type_str = self.return_type.to_java()

        body = [node.to_java() for node in self.body]

        # TODO: Implement these
        type_parameters = self.type_parameters
        throws = self.throws

        value = self.annotations_str + modifiers_str+return_type_str+" "+name_str+"("+parameters_str+")"

        value += "{" +  SEPERATOR + SEPERATOR.join(body) + SEPERATOR+"}"
        return value

class FieldDeclaration(Member, Declaration):
    """
        type: Node - Type(ReferenceType)
        declarators: List of Declarators
    """
    attrs = ("type", "declarators")

    def to_java(self):
        modifiers_str = self.modifiers_str
        type_str = self.type.to_java()

        declarators = [i.to_java() for i in self.declarators]

        value = self.annotations_str + modifiers_str+type_str+" "
        value += ", ".join(declarators)
        value += ";"
        
        return value

class ConstructorDeclaration(Declaration, Documented):
    attrs = ("type_parameters", "name", "parameters", "throws", "body")

    def to_java(self):
        modifiers_str = self.modifiers_str
        name_str = self.name
        parameters_str = get_parameter_str(self)

        body = [i.to_java() for i in self.body]

        value = self.annotations_str + modifiers_str+name_str+'('+parameters_str+')'
        value += " {" + SEPERATOR + SEPERATOR.join(body) + SEPERATOR+"}"
        return value
        
# ------------------------------------------------------------------------------

class ConstantDeclaration(FieldDeclaration):
    """
        only in interface
    """
    attrs = ()

class ArrayInitializer(Node):
    """
        different from ArrayCreator
        initializer: List of Node(Literals for example)
    """
    attrs = ("initializers",)

    def to_java(self):
        initialzers = [i.to_java() for i in self.initializers]

        value = "{" + ", ".join(initialzers) + "}"

        return value

class VariableDeclaration(Declaration):
    attrs = ("type", "declarators")
    
    def to_java(self):
        type_str = self.type.to_java()
        declarators = [i.to_java() for i in self.declarators]
        motifiers_str = self.modifiers_str
        annoations_str = self.annotations_str

        value = annoations_str + motifiers_str + type_str + " " + ", ".join(declarators)+";"

        return value

class LocalVariableDeclaration(VariableDeclaration):
    attrs = ()

    #def to_java(self):


class VariableDeclarator(Node):
    """
        name: String of name
        dimensions: List of (None or Node)
        initializer: 
    """
    attrs = ("name", "dimensions", "initializer")

    def to_java(self):
        name_and_dimensions_str = get_name_and_dimensions_str(self)
        value = name_and_dimensions_str
        if self.initializer:
            initializer_str = self.initializer.to_java()
            value += ' = '+initializer_str

        return value

class FormalParameter(Declaration):
    attrs = ("type", "name", "varargs")
    
    def to_java(self):
        type_str = self.type.to_java()
        value = self.modifiers_str+type_str+' '+self.name
        return value

class InferredFormalParameter(Node):
    attrs = ('name',)

# ------------------------------------------------------------------------------

class Statement(Node):
    attrs = ("label",)

    @property
    def label_str(self):
        if self.label:
            return self.label+": "
        else:
            return ""

class IfStatement(Statement):
    """
        condition: Node (BinaryOperation for example)
        then_statement: Node - BlockStatement or StatementExpression
        if_statement: Node - BlockStatement or StatementExpression 
    """
    attrs = ("condition", "then_statement", "else_statement")
    
    def to_java(self):
        condition_str = self.condition.to_java()
        then_statement_str = self.then_statement.to_java()
        else_statement_str = self.else_statement.to_java()

        value = self.label_str+"if("+condition_str+ ") "+SEPERATOR+then_statement_str+SEPERATOR+"else "+SEPERATOR+else_statement_str

        return value


class WhileStatement(Statement):
    """
        condition: Node (BinaryOperation for example)
        body: Node - BlockStatement or StatementExpression 
    """
    attrs = ("condition", "body")

    def to_java(self):
        condition_str = self.condition.to_java()
        body_str = self.body.to_java()

        value = self.label_str + "while ("+condition_str+") "+SEPERATOR+body_str
        return value

class DoStatement(Statement):
    """
        condition: Node (BinaryOperation for example)
        body: Node - BlockStatement or StatementExpression 
    """
    attrs = ("condition", "body")

    def to_java(self):
        condition_str = self.condition.to_java()
        body_str = self.body.to_java()
        value = self.label_str +"do"+SEPERATOR+body_str+  "while ("+condition_str+");"

        return value

class ForStatement(Statement):
    """
        control: Node - ForControl
        body: Node - BlockStatement or StatementExpression 
    """
    attrs = ("control", "body")

    def to_java(self):
        control_str = self.control.to_java()
        body_str = self.body.to_java()

        value = self.label_str+ "for ("+control_str+")"+SEPERATOR+body_str

        return value

class AssertStatement(Statement):
    attrs = ("condition", "value")

class BreakStatement(Statement):
    """
        goto: String - label of goto destination
    """
    attrs = ("goto",)

    def to_java(self):
        if self.goto:
            goto_str = ": "+self.goto
        else:
            goto_str = ""

        value = self.label_str + "break"+goto_str+";"
        return value
        

class ContinueStatement(Statement):
    """
        goto: String - label of goto destination
    """
    attrs = ("goto",)

    def to_java(self):
        if self.goto:
            goto_str = ": "+self.goto
        else:
            goto_str = ""

        value = self.label_str + "continue"+goto_str+";"
        return value

class ReturnStatement(Statement):
    """
        expression: Node - expression
    """
    attrs = ("expression",)
    
    def to_java(self):
        expression_str = self.expression.to_java()
        value = self.label_str+"return "+expression_str+";"

        return value

class ThrowStatement(Statement):
    attrs = ("expression",)

class SynchronizedStatement(Statement):
    attrs = ("lock", "block")

class TryStatement(Statement):
    attrs = ("resources", "block", "catches", "finally_block")

class SwitchStatement(Statement):
    attrs = ("expression", "cases")

class BlockStatement(Statement):
    """
        need to surround by {}
        statements: List of statements
    """
    attrs = ("statements",)

    def to_java(self):
        statements = [i.to_java() for i in self.statements]

        value = self.label_str+"{"+SEPERATOR+ SEPERATOR.join(statements)+SEPERATOR+"}"

        return value

class StatementExpression(Statement):
    """
        Complete statement need to add a ';'
        expression: Node - expression 
    """
    attrs = ("expression",)

    def to_java(self):
        expression_str = self.expression.to_java()
        
        ret = expression_str+";"
        return ret

# ------------------------------------------------------------------------------

class TryResource(Declaration):
    attrs = ("type", "name", "value")

class CatchClause(Statement):
    attrs = ("parameter", "block")

class CatchClauseParameter(Declaration):
    attrs = ("types", "name")

# ------------------------------------------------------------------------------

class SwitchStatementCase(Node):
    attrs = ("case", "statements")

class ForControl(Node):
    """
        init: List of Node(assignments) / Node LocalVariableDeclaration
        condition: Node
        update: List of Node(assignment)
    """
    attrs = ("init", "condition", "update")

    def to_java(self):
        if isinstance(self.init, list):
            inits_str = ", ".join([i.to_java() for i in self.init])+";"
        else:
            inits_str = self.init.to_java()
        condition_str = self.condition.to_java()
        updates = [i.to_java() for i in self.update]

        value = inits_str +condition_str+";"+",".join(updates)
        return value

class EnhancedForControl(Node):
    attrs = ("var", "iterable")

# ------------------------------------------------------------------------------

class Expression(Node):
    attrs = ()

class Assignment(Expression):
    """
        expressionl:
        value:
        type:
    """
    attrs = ("expressionl", "value", "type")

    def to_java(self):
        expressionl_str = self.expressionl.to_java()
        value_str = self.value.to_java()
        type_str = self.type

        ret = expressionl_str+' '+type_str+' '+value_str
        return ret

class TernaryExpression(Expression):
    attrs = ("condition", "if_true", "if_false")

class BinaryOperation(Expression):
    attrs = ("operator", "operandl", "operandr")

    def to_java(self):
        operator_str = self.operator
        operandl_str = self.operandl.to_java()
        operandr_str = self.operandr.to_java()

        value =  operandl_str + ' ' + operator_str + ' ' +operandr_str
        return value

class Cast(Expression):
    attrs = ("type", "expression")

class MethodReference(Expression):
    attrs = ("expression", "method", "type_arguments")

class LambdaExpression(Expression):
    attrs = ('parameters', 'body')

# ------------------------------------------------------------------------------

class Primary(Expression):
    attrs = ("prefix_operators", "postfix_operators", "qualifier", "selectors")

    @property
    def prefix_operators_str(self):
        return "".join(self.prefix_operators)

    @property
    def postfix_operators_str(self):
        return "".join(self.postfix_operators)

class Literal(Primary):
    """
        value: String of literal
    """
    attrs = ("value",)

    def to_java(self):
        value_str = self.value
        
        return value_str

class This(Primary):
    attrs = ()

    def to_java(self):
        return "this"

class MemberReference(Primary):
    attrs = ("member",)

    def to_java(self):
        member_str = self.member
        qualifier_str = self.qualifier

        #TODO:selector 
        if qualifier_str:
            value = qualifier_str+'.'+member_str
        else:
            value = member_str 
        
        
        value = self.prefix_operators_str+value+self.postfix_operators_str
        return value

class Invocation(Primary):
    attrs = ("type_arguments", "arguments")

class ExplicitConstructorInvocation(Invocation):
    attrs = ()

class SuperConstructorInvocation(Invocation):
    attrs = ()

    def to_java(self):
        arguments = [i.to_java() for i in self.arguments]
        value = "super("+ SEPERATOR.join(arguments) + ")"

        return value

class MethodInvocation(Invocation):
    attrs = ("member",)

class SuperMethodInvocation(Invocation):
    attrs = ("member",)

class SuperMemberReference(Primary):
    attrs = ("member",)

class ArraySelector(Expression):
    attrs = ("index",)

class ClassReference(Primary):
    attrs = ("type",)

class VoidClassReference(ClassReference):
    attrs = ()

# ------------------------------------------------------------------------------

class Creator(Primary):
    """
        create an object of a specific type
        type: Node - Type(ReferenceType)
    """
    attrs = ("type",)

    def to_java(self):
        type_str = self.type.to_java()
        value = "new "+type_str
        return value

class ArrayCreator(Creator):
    """
        dimensions: List of None or Node
        initializer: Node - ArrayInitializer
    """
    attrs = ("dimensions", "initializer")

    def to_java(self):
        type_str = self.type.to_java()
        type_and_dimension_str = get_name_and_dimensions_str_bare(type_str, self.dimensions)
        if self.initializer:
            initializer_str = self.initializer.to_java()
        else:
            initializer_str = ""

        value = "new "+type_and_dimension_str+initializer_str

        return value


class ClassCreator(Creator):
    """
        constructor_type_arguments: ? 
        arguments: List of Node (Literals for example)
        body: List of Node
    """
    attrs = ("constructor_type_arguments", "arguments", "body")

    def to_java(self):
        type_str = self.type.to_java()

        if self.body != None:
            body = [i.to_java() for i in self.body]
            body_str = "{" + SEPERATOR.join(body) + "}"
        else:
            body_str = ""

        arguments = [i.to_java() for i in self.arguments]
        arguments_str = ", ".join(arguments)
        
        value = "new "+type_str+"("+arguments_str+")"+body_str
        return value

class InnerClassCreator(Creator):
    attrs = ("constructor_type_arguments", "arguments", "body")

# ------------------------------------------------------------------------------

class EnumBody(Node):
    attrs = ("constants", "declarations")

class EnumConstantDeclaration(Declaration, Documented):
    attrs = ("name", "arguments", "body")

class AnnotationMethod(Declaration):
    attrs = ("name", "return_type", "dimensions", "default")


def get_name_and_dimensions_str_bare(name, dimensions):
    ret = name
    for i in dimensions:
        if i:
            ret += "["+i.to_java()+"]"
        else:
            ret += "[]"
        
    return ret

#TODO: Modify dimension expansion
def get_name_and_dimensions_str(node):
    if not node.dimensions:
        return node.name
    return get_name_and_dimensions_str_bare(node.name, node.dimensions)

def get_parameter_str(node):
    if node.parameters:
        paras = []
        for para in node.parameters:
            paras.append(para.to_java())

        return ", ".join(paras)
    else:
        return ""
