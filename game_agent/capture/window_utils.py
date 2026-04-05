import ctypes
import ctypes.wintypes
from typing import List, Tuple, Optional


def find_window_rect(title: str, class_name: str = None) -> Tuple[int, int, int, int]:
    """Find a window by title and return (left, top, width, height).

    If class_name is provided, searches for a window with that class name first.
    Otherwise enumerates all windows to find one matching the title with a
    pygame/SDL class (to avoid matching console windows).
    """
    # Try pygame class first, then SDL, then fallback
    found_hwnd = None
    for cls_try in ["pygame", "SDL_app", None]:
        hwnd = ctypes.windll.user32.FindWindowW(cls_try, title)
        if hwnd:
            found_hwnd = hwnd
            break

    if not found_hwnd:
        available = list_windows()
        raise RuntimeError(
            f"Window '{title}' not found. Available windows:\n"
            + "\n".join(f"  - {w}" for w in available[:20])
        )

    # Use GetClientRect + ClientToScreen to get the actual content area
    # (excludes window border and title bar)
    client_rect = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetClientRect(found_hwnd, ctypes.byref(client_rect))

    # Convert client top-left to screen coordinates
    point = ctypes.wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(found_hwnd, ctypes.byref(point))

    left = point.x
    top = point.y
    width = client_rect.right
    height = client_rect.bottom
    return (left, top, width, height)


def set_foreground(title: str):
    """Bring a window to the foreground."""
    # Re-use find_window_rect's enumeration to get the right HWND
    rect = find_window_rect(title)
    # Find the hwnd again for SetForegroundWindow
    hwnd = ctypes.windll.user32.FindWindowW("pygame", title)
    if not hwnd:
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
    if hwnd:
        ctypes.windll.user32.SetForegroundWindow(hwnd)


def list_windows() -> List[str]:
    """List all visible window titles."""
    titles: List[str] = []

    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def enum_callback(hwnd, _lparam):
        if ctypes.windll.user32.IsWindowVisible(hwnd):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                titles.append(buf.value)
        return True

    ctypes.windll.user32.EnumWindows(enum_callback, 0)
    return sorted(titles)
