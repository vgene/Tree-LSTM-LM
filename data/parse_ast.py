import tqdm
import javalang
import pickle
import os
import time

EXPRFILE = "expr.pkl"
INDEXFILE = "index.pkl"
DATAPATH = "java_projects"
ASTPATH = "java_projects_ast"

print("Reading files")
expr = pickle.load(open(EXPRFILE, 'rb'))
index = pickle.load(open(INDEXFILE, 'rb'))

categories = ['train', 'test', 'val']
data={}

# Transform text code into ASTs
for cat in categories:
    asts = []
    for idx, project in enumerate(expr[cat]):
        print("%d of %d, Project: %s"%(idx+1, len(expr[cat]), project))
        for fname in tqdm.tqdm(index[project]):
            with open(os.path.join(DATAPATH, fname)) as f:
                code = f.read()
            try:
                ast = javalang.parse.parse(code)
            except Exception as e: #javalang.parser.JavaSyntaxError as e:
                print("Failed:"+fname)
            asts.append(ast)
    data[cat] = asts

print("Writing training data")
pickle.dump(data, open(os.path.join(ASTPATH, time.strftime("%b%d%H"+".pkl")), 'wb'))
