import importlib
import os
import json
from pathlib import Path
from tree_sitter import Query, QueryCursor
from retriever.parser_setup import get_parsers, get_languages

# Mapping of file extensions to supported languages
EXTENSION_TO_LANGUAGE = {
    '.py': 'python',
    '.java': 'java',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.h': 'cpp',
    '.hpp': 'cpp',
    '.cs': 'c_sharp'
}

PACKAGE_QUERY_DIRS = {
    'python': 'tree_sitter_python',
    'java': 'tree_sitter_java',
    'javascript': 'tree_sitter_javascript',
    'cpp': 'tree_sitter_cpp',
    'c_sharp': 'tree_sitter_c_sharp',
}

def packaged_tags_path(lang):
    module = importlib.import_module(PACKAGE_QUERY_DIRS[lang])
    package_root = Path(module.__file__).parent
    # tree-sitter-c-sharp installs its queries under a hyphenated sibling directory.
    candidates = [package_root / "queries" / "tags.scm",
                  package_root.parent / "tree_sitter_c-sharp" / "queries" / "tags.scm"]
    return next((p for p in candidates if p.exists()), None)

def load_queries(languages):
    """Loads tag queries for all supported languages."""
    queries = {}
    for lang in languages:
        folder_name = lang.replace('_', '-')
        query_path = Path(f"retriever/grammars/tree-sitter-{folder_name}/queries/tags.scm")
        if not query_path.exists():
            query_path = packaged_tags_path(lang) or query_path

        if query_path.exists():
            with open(query_path, 'r') as f:
                queries[lang] = f.read()
        else:
            print(f"Warning: No tags.scm found for {lang} at {query_path}")
            
    return queries

def index_codebase(root_dir):
    """
    Traverses the codebase, parses supported source files, and extracts definitions using Query matches.
    """
    parsers = get_parsers()
    languages = get_languages()
    query_scms = load_queries(languages.keys())
    
    # Pre-compile queries
    queries = {}
    for lang, scm in query_scms.items():
        try:
            queries[lang] = Query(languages[lang], scm)
        except Exception as e:
            print(f"Error compiling query for {lang}: {e}")
    
    index = []
    
    # Directories to ignore
    IGNORE_DIRS = {'.git', '.venv', 'venv', 'bak_venv', '__pycache__', 'node_modules', 'build', 'dist', 'grammars'}
    
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
        
        for file in files:
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            
            if ext not in EXTENSION_TO_LANGUAGE:
                continue
                
            lang = EXTENSION_TO_LANGUAGE[ext]
            if lang not in parsers or lang not in queries:
                continue
            
            parser = parsers[lang]
            query = queries[lang]
            
            print(f"Indexing {file_path} ({lang})...")
            
            try:
                with open(file_path, 'rb') as f:
                    code = f.read()
                
                tree = parser.parse(code)
                cursor = QueryCursor(query)
                
                # In 0.25.2, matches(node) returns list[tuple[int, dict[str, list[Node]]]]
                matches = cursor.matches(tree.root_node)
                
                for pattern_idx, captures in matches:
                    # Captures is a dict mapping tag name to a list of nodes
                    # Find the definition tag and the name tag
                    definition_tag = next((t for t in captures.keys() if t.startswith('definition.')), None)
                    
                    if definition_tag and 'name' in captures:
                        def_node = captures[definition_tag][0]
                        name_node = captures['name'][0]
                        
                        name = code[name_node.start_byte:name_node.end_byte].decode('utf8', errors='ignore')
                        
                        index.append({
                            'name': name,
                            'type': definition_tag.split('.')[-1],
                            'language': lang,
                            'file_path': str(file_path),
                            'start_line': def_node.start_point[0] + 1,
                            'end_line': def_node.end_point[0] + 1,
                            'start_column': def_node.start_point[1],
                            'end_column': def_node.end_point[1]
                        })
                        
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
                
    return index

if __name__ == "__main__":
    PROJECT_ROOT = "."
    OUTPUT_FILE = "data/processed/ast_index.json"
    
    os.makedirs("data/processed", exist_ok=True)
    
    print("Starting AST indexing...")
    full_index = index_codebase(PROJECT_ROOT)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(full_index, f, indent=2)
        
    print(f"Indexing complete. {len(full_index)} definitions stored in {OUTPUT_FILE}")
