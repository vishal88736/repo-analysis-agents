"""
Tree-sitter based static code parser.

Supports: Python, JavaScript, TypeScript, Go, Java, Rust, C, C++
Extracts: functions, classes, methods, imports
"""

import logging
from typing import Any

from tree_sitter import Language, Parser, Node

from app.schemas.analysis import ParsedStructure, ParsedFunction, ParsedClass, ParsedImport

logger = logging.getLogger(__name__)

# Language registry: extension → (module_import_path, language_func)
# We dynamically import to avoid hard failures if a grammar isn't installed
LANGUAGE_MAP: dict[str, str] = {
    ".py": "tree_sitter_python",
    ".js": "tree_sitter_javascript",
    ".jsx": "tree_sitter_javascript",
    ".ts": "tree_sitter_typescript",
    ".tsx": "tree_sitter_typescript",
    ".go": "tree_sitter_go",
    ".java": "tree_sitter_java",
    ".rs": "tree_sitter_rust",
    ".c": "tree_sitter_c",
    ".cpp": "tree_sitter_cpp",
    ".cc": "tree_sitter_cpp",
    ".h": "tree_sitter_c",
    ".hpp": "tree_sitter_cpp",
}

# Cache loaded languages and parsers
_language_cache: dict[str, Language] = {}
_parser_cache: dict[str, Parser] = {}


def _get_parser(extension: str) -> Parser | None:
    """Get or create a tree-sitter parser for the given file extension."""
    if extension in _parser_cache:
        return _parser_cache[extension]

    module_name = LANGUAGE_MAP.get(extension)
    if not module_name:
        return None

    try:
        import importlib
        mod = importlib.import_module(module_name)
        lang = Language(mod.language())
        parser = Parser(lang)
        _language_cache[extension] = lang
        _parser_cache[extension] = parser
        return parser
    except ImportError:
        logger.warning("Tree-sitter grammar not installed: %s (for %s)", module_name, extension)
        return None
    except Exception as e:
        logger.warning("Failed to initialize tree-sitter for %s: %s", extension, e)
        return None


def _get_node_text(node: Node, source: bytes) -> str:
    """Extract text from a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_python(root: Node, source: bytes) -> ParsedStructure:
    """Extract structure from Python source."""
    functions: list[ParsedFunction] = []
    classes: list[ParsedClass] = []
    imports: list[ParsedImport] = []

    for child in root.children:
        if child.type == "function_definition":
            name_node = child.child_by_field_name("name")
            params_node = child.child_by_field_name("parameters")
            params = []
            if params_node:
                for p in params_node.children:
                    if p.type == "identifier":
                        params.append(_get_node_text(p, source))
            functions.append(ParsedFunction(
                name=_get_node_text(name_node, source) if name_node else "unknown",
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                parameters=params,
            ))

        elif child.type == "class_definition":
            name_node = child.child_by_field_name("name")
            class_name = _get_node_text(name_node, source) if name_node else "unknown"
            methods = []
            body = child.child_by_field_name("body")
            if body:
                for stmt in body.children:
                    if stmt.type == "function_definition":
                        m_name = stmt.child_by_field_name("name")
                        if m_name:
                            methods.append(_get_node_text(m_name, source))
            classes.append(ParsedClass(
                name=class_name,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                methods=methods,
            ))

        elif child.type == "import_statement":
            text = _get_node_text(child, source)
            parts = text.replace("import ", "").strip().split(",")
            imports.append(ParsedImport(module=parts[0].strip(), names=[p.strip() for p in parts]))

        elif child.type == "import_from_statement":
            text = _get_node_text(child, source)
            # from X import Y, Z
            if " import " in text:
                module_part = text.split(" import ")[0].replace("from ", "").strip()
                names_part = text.split(" import ")[1].strip()
                names = [n.strip() for n in names_part.split(",")]
                imports.append(ParsedImport(module=module_part, names=names))

    return ParsedStructure(functions=functions, classes=classes, imports=imports)


def _extract_js_ts(root: Node, source: bytes) -> ParsedStructure:
    """Extract structure from JavaScript/TypeScript source."""
    functions: list[ParsedFunction] = []
    classes: list[ParsedClass] = []
    imports: list[ParsedImport] = []

    def walk(node: Node):
        if node.type in ("function_declaration", "method_definition"):
            name_node = node.child_by_field_name("name")
            params_node = node.child_by_field_name("parameters")
            params = []
            if params_node:
                for p in params_node.children:
                    if p.type in ("identifier", "required_parameter", "optional_parameter"):
                        params.append(_get_node_text(p, source))
            functions.append(ParsedFunction(
                name=_get_node_text(name_node, source) if name_node else "anonymous",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parameters=params,
            ))

        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            class_name = _get_node_text(name_node, source) if name_node else "unknown"
            methods = []
            body = node.child_by_field_name("body")
            if body:
                for stmt in body.children:
                    if stmt.type == "method_definition":
                        m_name = stmt.child_by_field_name("name")
                        if m_name:
                            methods.append(_get_node_text(m_name, source))
            classes.append(ParsedClass(
                name=class_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                methods=methods,
            ))

        elif node.type == "import_statement":
            text = _get_node_text(node, source)
            # Simple extraction: from '...'
            source_node = node.child_by_field_name("source")
            module = _get_node_text(source_node, source).strip("'\"") if source_node else text
            imports.append(ParsedImport(module=module, names=[]))

        for child in node.children:
            walk(child)

    walk(root)
    return ParsedStructure(functions=functions, classes=classes, imports=imports)


def _extract_generic(root: Node, source: bytes) -> ParsedStructure:
    """Generic extraction for Go, Java, Rust, C, C++."""
    functions: list[ParsedFunction] = []
    classes: list[ParsedClass] = []

    def walk(node: Node):
        if node.type in (
            "function_declaration", "function_definition",
            "method_declaration", "function_item",
        ):
            name_node = node.child_by_field_name("name")
            functions.append(ParsedFunction(
                name=_get_node_text(name_node, source) if name_node else "unknown",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))

        elif node.type in ("class_declaration", "struct_item", "struct_specifier", "impl_item"):
            name_node = node.child_by_field_name("name")
            classes.append(ParsedClass(
                name=_get_node_text(name_node, source) if name_node else "unknown",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))

        for child in node.children:
            walk(child)

    walk(root)
    return ParsedStructure(functions=functions, classes=classes)


# Dispatcher
_EXTRACTORS: dict[str, Any] = {
    ".py": _extract_python,
    ".js": _extract_js_ts,
    ".jsx": _extract_js_ts,
    ".ts": _extract_js_ts,
    ".tsx": _extract_js_ts,
}


def parse_file(file_path: str, content: bytes, extension: str) -> ParsedStructure:
    """
    Parse a source file using tree-sitter and return extracted structure.

    Args:
        file_path: Relative path (for logging).
        content: Raw file bytes.
        extension: File extension including dot (e.g. ".py").

    Returns:
        ParsedStructure with functions, classes, and imports.
    """
    parser = _get_parser(extension)
    if parser is None:
        logger.debug("No tree-sitter parser for %s, returning empty structure.", extension)
        return ParsedStructure()

    try:
        tree = parser.parse(content)
        root = tree.root_node

        extractor = _EXTRACTORS.get(extension, _extract_generic)
        result = extractor(root, content)
        logger.debug(
            "Parsed %s: %d functions, %d classes, %d imports",
            file_path,
            len(result.functions),
            len(result.classes),
            len(result.imports),
        )
        return result

    except Exception as e:
        logger.warning("Tree-sitter parsing failed for %s: %s", file_path, e)
        return ParsedStructure()