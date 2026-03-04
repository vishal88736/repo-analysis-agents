"""Repository Scanner — clones GitHub repo, scans files recursively."""

import logging
import os
from pathlib import Path

import git

from app.config import settings
from app.core.exceptions import RepositoryCloneError
from app.schemas.analysis import FileMetadata, RepoMap, RepoFileInfo

logger = logging.getLogger(__name__)

IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", ".env",
    "dist", "build", "__pycache__", ".tox", ".mypy_cache",
    ".pytest_cache", "vendor", "target", ".idea", ".vscode",
    "coverage", ".next", ".nuxt", "out", ".cache",
    ".parcel-cache", "bower_components",
}

# Feature 6: Enhanced file filtering
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".map", ".min.js", ".min.css", ".lock",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".zip", ".tar", ".gz",
    ".pdf", ".exe", ".dll", ".so", ".dylib",
}

IGNORED_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java",
    ".rs", ".c", ".cpp", ".cc", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".scala",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
    ".md", ".rst", ".txt", ".sql", ".graphql", ".proto",
    ".sh", ".bash", ".zsh", ".dockerfile", ".tf", ".hcl",
}

LANG_MAP = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".go": "Go",
    ".java": "Java", ".rs": "Rust", ".c": "C", ".cpp": "C++",
    ".cc": "C++", ".h": "C", ".hpp": "C++", ".rb": "Ruby",
    ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin",
    ".scala": "Scala", ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON", ".toml": "TOML", ".md": "Markdown",
    ".sql": "SQL", ".sh": "Shell", ".dockerfile": "Dockerfile",
}


def _repo_dir_name(url: str) -> str:
    parts = url.rstrip("/").rstrip(".git").split("/")
    return f"{parts[-2]}__{parts[-1]}" if len(parts) >= 2 else parts[-1]


def clone_repository(url: str) -> Path:
    """Clone (shallow) or reuse cached repo."""
    dir_name = _repo_dir_name(url)
    clone_path = settings.clone_path / dir_name

    if clone_path.exists() and (clone_path / ".git").exists():
        logger.info("Using cached clone: %s", clone_path)
        try:
            repo = git.Repo(clone_path)
            repo.remotes.origin.pull()
        except Exception as e:
            logger.warning("Pull failed, using cache: %s", e)
        return clone_path

    try:
        logger.info("Cloning: %s → %s", url, clone_path)
        git.Repo.clone_from(url, str(clone_path), depth=1, single_branch=True)
        return clone_path
    except git.GitCommandError as e:
        raise RepositoryCloneError(f"Clone failed: {url}", details=str(e))


def scan_repository(repo_path: Path) -> list[FileMetadata]:
    """Recursively scan repo for code files."""
    files = []
    max_size = settings.max_file_size_kb * 1024

    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for filename in filenames:
            # Feature 6: Skip ignored files by name
            if filename in IGNORED_FILES:
                continue

            full_path = Path(root) / filename
            ext = full_path.suffix.lower()
            if filename.lower() == "dockerfile":
                ext = ".dockerfile"

            # Feature 6: Skip ignored extensions
            if ext in IGNORED_EXTENSIONS:
                continue

            if ext not in CODE_EXTENSIONS:
                continue

            try:
                size = full_path.stat().st_size
            except OSError:
                continue
            if size > max_size or size == 0:
                continue

            files.append(FileMetadata(
                path=str(full_path.relative_to(repo_path)),
                size_bytes=size,
                language=LANG_MAP.get(ext, "unknown"),
                extension=ext,
            ))

    logger.info("Scanned %d files in %s", len(files), repo_path)
    return files


def build_repo_map(files: list[FileMetadata]) -> RepoMap:
    """Feature 1: Build a repo map from scanned file metadata."""
    repo_files: dict[str, RepoFileInfo] = {}
    total_tokens = 0

    for f in files:
        # Estimate tokens: ~4 chars per token
        tokens_estimate = f.size_bytes // 4
        total_tokens += tokens_estimate
        repo_files[f.path] = RepoFileInfo(
            language=f.language,
            size=f.size_bytes,
            tokens_estimate=tokens_estimate,
            file_type=f.extension,
        )

    return RepoMap(
        files=repo_files,
        total_files=len(files),
        total_tokens_estimate=total_tokens,
    )
