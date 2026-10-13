import ast
worker_source = (ROOT / 'workshop.py').read_text(encoding='utf-8')
for node in ast.parse(worker_source).body:
    if isinstance(node, ast.FunctionDef) and node.name in ('support', 'aligned_values', 'score'):
        print(ast.get_source_segment(worker_source, node), end='\n\n')
