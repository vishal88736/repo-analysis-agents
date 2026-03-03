"""
Repository Scanner — clones a GitHub repo and scans all files.
"""

import logging
import os
from pathlib import Path

import git

from app.config import settings
from app.core.exceptions import RepositoryCloneError
from app.schemas.analysis import FileMetadata

logger = logging.getLogger(__name__)

# Directories to skip
IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", ".env",
    "dist", "build", "__pycache__", ".tox", ".mypy_cache",
    ".pytest_cache", "vendor", "target", ".idea", ".vscode",
    "coverage", ".next", ".nuxt", "out",
}

# File extensions we care about
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java",
    ".rs", ".c", ".cpp", ".cc", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".scala",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
    ".md", ".rst", ".txt",
    ".sql", ".graphql", ".proto",
    ".sh", ".bash", ".zsh",
    ".dockerfile", ".tf", ".hcl",
}

LANGUAGE_MAP = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".go": "Go",
    ".java": "Java", ".rs": "Rust", ".c": "C", ".cpp": "C++",
    ".cc": "C++", ".h": "C", ".hpp": "C++", ".rb": "Ruby",
    ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin",
    ".scala": "Scala", ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON", ".toml": "TOML", ".ini": "INI",
    ".cfg": "Config", ".md": "Markdown", ".rst": "reStructuredText",
    ".txt": "Text", ".sql": "SQL", ".graphql": "GraphQL",
    ".proto": "Protobuf", ".sh": "Shell", ".bash": "Shell",
    ".zsh": "Shell", ".dockerfile": "Dockerfile",
    ".tf": "Terraform", ".hcl": "HCL",
}


def _repo_dir_name(url: str) -> str:
    """Create a directory name from a GitHub URL."""
    # https://github.com/owner/repo.git → owner__repo
    parts = url.rstrip("/").rstrip(".git").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}__{parts[-1]}"
    return parts[-1]


def clone_repository(url: str) -> Path:
    """
    Clone a GitHub repository to local storage.
    Returns the path to the cloned directory.
    Reuses existing clone if present (cache).
    """
    dir_name = _repo_dir_name(url)
    clone_path = settings.clone_path / dir_name

    if clone_path.exists() and (clone_path / ".git").exists():
        logger.info("Repository already cloned: %s", clone_path)
        # Pull latest
        try:
            repo = git.Repo(clone_path)
            repo.remotes.origin.pull()
            logger.info("Pulled latest changes for: %s", url)
        except Exception as e:
            logger.warning("Failed to pull latest, using cached: %s", e)
        return clone_path

    try:
        logger.info("Cloning repository: %s → %s", url, clone_path)
        git.Repo.clone_from(
            url,
            str(clone_path),
            depth=1,  # Shallow clone for speed
            single_branch=True,
        )
        logger.info("Clone completed: %s", url)
        return clone_path
    except git.GitCommandError as e:
        raise RepositoryCloneError(
            f"Failed to clone repository: {url}",
            details=str(e),
        )


def scan_repository(repo_path: Path) -> list[FileMetadata]:
    """
    Recursively scan a cloned repository and return metadata for all code files.
    """
    files: list[FileMetadata] = []
    max_size = settings.max_file_size_kb * 1024

    for root, dirs, filenames in os.walk(repo_path):
        # Filter out ignored directories (in-place modification)
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for filename in filenames:
            full_path = Path(root) / filename
            ext = full_path.suffix.lower()

            # Special case: Dockerfile without extension
            if filename.lower() == "dockerfile":
                ext = ".dockerfile"

            if ext not in CODE_EXTENSIONS:
                continue

            try:
                size = full_path.stat().st_size
            except OSError:
                continue

            if size > max_size:
                logger.debug("Skipping large file: %s (%d KB)", full_path, size // 1024)
                continue

            if size == 0:
                continue

            rel_path = str(full_path.relative_to(repo_path))
            language = LANGUAGE_MAP.get(ext, "unknown")

            files.append(FileMetadata(
                path=rel_path,
                size_bytes=size,
                language=language,
                extension=ext,
            ))

    logger.info("Scanned %d files in %s", len(files), repo_path)
    return files