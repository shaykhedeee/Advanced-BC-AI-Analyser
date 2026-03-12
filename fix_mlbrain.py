"""Fix corrupted enhanced_ml_brain.py"""
path = r'c:\Users\USER\Documents\AI CRACKER\edge_tracker\enhanced_ml_brain.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix literal \n sequences
content = content.replace('\\n', '\n')

# Fix escaped triple quotes in docstrings  
content = content.replace('\\"\\"\\"', '"""')

# Fix remaining escaped double quotes (but preserve Python string escaping)
# Only fix \" that are standalone docstring markers
import re
content = re.sub(r'\\"([^"\\])', r'"\1', content)
content = re.sub(r'([^"\\])\\"', r'\1"', content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done. Lines: {content.count(chr(10))}")

# Verify syntax
import ast
try:
    ast.parse(content)
    print("SYNTAX OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR at line {e.lineno}: {e.msg}")
    print(f"Context: {content.splitlines()[e.lineno-1] if e.lineno else 'N/A'}")
