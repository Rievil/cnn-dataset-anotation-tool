from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PySide6.QtCore import QObject, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

DOCUMENTATION_URL = "https://github.com/Rievil/cnn-dataset-anotation-tool/blob/main/docs/app_design.md"


@dataclass
class GitVersionInfo:
    repo_root: Path
    local_commit: str
    remote_commit: Optional[str]
    branch: Optional[str]
    remote_name: str
    remote_ref: str

    @property
    def short_local(self) -> str:
        return self.local_commit[:8]

    @property
    def short_remote(self) -> str:
        return self.remote_commit[:8] if self.remote_commit else "Unknown"

    @property
    def tracking_ref(self) -> str:
        ref = self.remote_ref or "HEAD"
        return f"{self.remote_name}/{ref}"

    @property
    def branch_label(self) -> str:
        return self.branch or "Detached HEAD"

    @property
    def update_available(self) -> bool:
        return bool(self.remote_commit and self.remote_commit != self.local_commit)


class FunctionWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, func: Callable[[], Any]) -> None:
        super().__init__()
        self._func = func

    def run(self) -> None:
        try:
            result = self._func()
        except Exception as exc:  # pragma: no cover - surfaced in UI
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


def _find_git_root(start: Optional[Path] = None) -> Optional[Path]:
    current = start or Path(__file__).resolve()
    for path in (current, *current.parents):
        if (path / ".git").exists():
            return path
    return None


def _run_git_command(repo_root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Unknown error"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _fetch_git_version_info() -> GitVersionInfo:
    repo_root = _find_git_root(Path(__file__).resolve().parent)
    if repo_root is None:
        raise RuntimeError("Git repository not found in this installation.")

    local_commit = _run_git_command(repo_root, ["rev-parse", "HEAD"])

    branch = _run_git_command(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch == "HEAD":
        branch = None

    remote_name = "origin"
    remote_ref: Optional[str] = None
    try:
        upstream = _run_git_command(
            repo_root,
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        )
        if upstream:
            if "/" in upstream:
                remote_name, remote_ref = upstream.split("/", 1)
            else:
                remote_ref = upstream
    except RuntimeError:
        remote_ref = branch or "HEAD"

    if not remote_ref:
        remote_ref = "HEAD"

    ls_output = _run_git_command(repo_root, ["ls-remote", remote_name, remote_ref])
    remote_commit = None
    for line in ls_output.splitlines():
        parts = line.split()
        if parts:
            remote_commit = parts[0]
            break

    if remote_commit is None:
        raise RuntimeError(f"Unable to determine latest commit for {remote_name}/{remote_ref}.")

    return GitVersionInfo(
        repo_root=repo_root,
        local_commit=local_commit,
        remote_commit=remote_commit,
        branch=branch,
        remote_name=remote_name,
        remote_ref=remote_ref,
    )


def _pull_latest_changes(repo_root: Path) -> str:
    return _run_git_command(repo_root, ["pull", "--ff-only"])


class AboutDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About CNN Dataset Annotation Tool")
        self._version_info: Optional[GitVersionInfo] = None
        self._version_thread: Optional[QThread] = None
        self._version_worker: Optional[FunctionWorker] = None
        self._update_thread: Optional[QThread] = None
        self._update_worker: Optional[FunctionWorker] = None
        self._build_ui()
        self._start_version_check()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("<h2>CNN Dataset Annotation Tool</h2>")
        layout.addWidget(title)

        description = QLabel(
            "Manage RGB/label pairs for CNN datasets, export training tiles, and keep your tooling up to date."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        doc_button = QPushButton("Open Documentation")
        doc_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOCUMENTATION_URL)))
        layout.addWidget(doc_button)

        version_group = QGroupBox("Version & Updates")
        form = QFormLayout(version_group)
        self.local_version_label = QLabel("Detecting…")
        self.remote_version_label = QLabel("—")
        self.status_label = QLabel("Checking for updates…")
        form.addRow("Current commit:", self.local_version_label)
        form.addRow("Latest remote:", self.remote_version_label)
        form.addRow("Status:", self.status_label)
        layout.addWidget(version_group)

        self.update_button = QPushButton("Get newest version")
        self.update_button.hide()
        self.update_button.clicked.connect(self._start_update)
        layout.addWidget(self.update_button)

        self.update_status_label = QLabel()
        self.update_status_label.setWordWrap(True)
        self.update_status_label.hide()
        layout.addWidget(self.update_status_label)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

    def _start_version_check(self, *, reset_update_status: bool = True) -> None:
        self.status_label.setText("Checking for updates…")
        self.remote_version_label.setText("—")
        self.local_version_label.setText("Detecting…")
        self.update_button.hide()
        if reset_update_status:
            self.update_status_label.hide()
        self._run_in_thread(
            _fetch_git_version_info,
            self._on_version_success,
            self._on_version_error,
            version_worker=True,
        )

    def _start_update(self) -> None:
        if not self._version_info:
            return
        self.update_button.setEnabled(False)
        self.update_status_label.setText("Fetching latest changes from GitHub…")
        self.update_status_label.show()
        repo_root = self._version_info.repo_root
        self._run_in_thread(
            lambda: _pull_latest_changes(repo_root),
            self._on_update_success,
            self._on_update_error,
            version_worker=False,
        )

    def _run_in_thread(
        self,
        func: Callable[[], Any],
        success_cb: Callable[[Any], None],
        error_cb: Callable[[str], None],
        *,
        version_worker: bool,
    ) -> None:
        worker = FunctionWorker(func)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(success_cb)
        worker.failed.connect(error_cb)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        if version_worker:
            self._version_worker = worker
            self._version_thread = thread
        else:
            self._update_worker = worker
            self._update_thread = thread

        thread.start()

    def _on_version_success(self, payload: Any) -> None:
        self._version_worker = None
        self._version_thread = None
        info = payload
        if not isinstance(info, GitVersionInfo):
            self._on_version_error("Invalid version information received.")
            return
        self._version_info = info
        branch = info.branch_label
        self.local_version_label.setText(f"{info.short_local} ({branch})")
        self.remote_version_label.setText(f"{info.short_remote} ({info.tracking_ref})")
        if info.update_available:
            self.status_label.setText("A newer version is available.")
            self.update_button.setEnabled(True)
            self.update_button.show()
        else:
            self.status_label.setText("You are running the latest version.")
            self.update_button.hide()

    def _on_version_error(self, message: str) -> None:
        self._version_worker = None
        self._version_thread = None
        self._version_info = None
        self.status_label.setText(message)
        self.remote_version_label.setText("Unavailable")
        self.update_button.hide()

    def _on_update_success(self, payload: Any) -> None:
        self._update_worker = None
        self._update_thread = None
        output = str(payload).strip()
        summary = output.splitlines()[0] if output else "Repository updated."
        self.update_status_label.setText(
            f"Update complete: {summary}\nPlease restart the application to use the newest code."
        )
        self.update_button.setEnabled(True)
        self._start_version_check(reset_update_status=False)

    def _on_update_error(self, message: str) -> None:
        self._update_worker = None
        self._update_thread = None
        self.update_status_label.setText(f"Update failed: {message}")
        self.update_button.setEnabled(True)
