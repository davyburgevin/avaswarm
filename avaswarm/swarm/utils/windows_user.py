import os
import getpass
import subprocess
import ctypes
from ctypes import create_unicode_buffer, byref


def get_windows_display_name() -> str | None:
    """Attempt to retrieve the user's display name on Windows.

    Tries GetUserNameExW, then PowerShell query, then environment fallback.
    """
    # Try Win32 API GetUserNameExW
    try:
        NameDisplay = 3  # NameDisplay
        buf = create_unicode_buffer(256)
        size = ctypes.c_ulong(len(buf))
        res = ctypes.windll.secur32.GetUserNameExW(NameDisplay, buf, byref(size))
        if res:
            name = buf.value.strip()
            if name:
                return name
    except Exception:
        pass

    # Try PowerShell to fetch FullName from Win32_UserAccount
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance -ClassName Win32_UserAccount -Filter \"Name = '$env:USERNAME'\").FullName",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        out = p.stdout.strip()
        if out:
            return out
    except Exception:
        pass

    # Fallbacks
    uname = os.environ.get("USERNAME") or getpass.getuser()
    return uname or None


def split_name(display_name: str) -> tuple[str | None, str | None]:
    if not display_name:
        return None, None
    parts = display_name.strip().split()
    if len(parts) == 1:
        return parts[0], None
    return parts[0], " ".join(parts[1:])
