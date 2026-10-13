"""
Single-instance guard for the Telegram bot.

Telegram allows exactly one getUpdates consumer per bot token. When two copies
of the bot run at the same time they constantly cancel each other's long-poll
requests and both start answering with "409 Conflict: terminated by other
getUpdates request".

This module grabs an OS-level named mutex, which the kernel releases
automatically when the process dies (even on a hard kill / crash). The second
copy therefore refuses to start with a readable message instead of flooding the
log with 409 errors.
"""

from __future__ import annotations

import ctypes
import hashlib
import sys

# winerror.h: ERROR_ALREADY_EXISTS
_ERROR_ALREADY_EXISTS = 183


class SingleInstanceError(RuntimeError):
    """Another copy of this application is already running."""


def lock_name_for(token: str) -> str:
    """
    Build a mutex name that is unique per bot token, so different bots (and
    different projects) never block each other.
    """
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return f"Local\\PineconeManagerBot_{digest}"


def acquire(token: str) -> int | None:
    """
    Try to become the only running instance.

    Returns a handle that must stay referenced for the whole process lifetime
    (the caller keeps it in a module-level variable), or None on platforms
    without a Windows named mutex.

    Raises SingleInstanceError if the lock is already held.
    """
    if sys.platform != "win32":
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_bool

    name = lock_name_for(token)
    handle = kernel32.CreateMutexW(None, False, name)
    last_error = ctypes.get_last_error()

    if not handle:
        raise OSError(last_error, f"CreateMutexW failed for {name}")

    if last_error == _ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        raise SingleInstanceError(
            "Другой экземпляр этого бота уже запущен и забрал getUpdates. "
            "Telegram допускает только один опрос на токен — второй процесс "
            "получает 409 Conflict. Закрой старый (см. stop_bot.ps1) и запусти снова."
        )

    return handle