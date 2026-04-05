import ctypes
import ctypes.wintypes
import numpy as np
from typing import Optional, Tuple


class FrameGrabber:
    """Captures a window's content using Win32 PrintWindow API.

    Works even when the window is behind other windows or partially covered.
    """

    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None,
                 window_title: Optional[str] = None):
        if window_title is not None:
            # Find the pygame window handle directly
            self.hwnd = ctypes.windll.user32.FindWindowW("pygame", window_title)
            if not self.hwnd:
                self.hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
            if not self.hwnd:
                raise RuntimeError(f"Window '{window_title}' not found")

            # Get client size
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetClientRect(self.hwnd, ctypes.byref(rect))
            self.width = rect.right
            self.height = rect.bottom
        elif region is not None:
            self.hwnd = None
            self.width = region[2]
            self.height = region[3]
            self._region = region
        else:
            raise ValueError("Provide either region or window_title.")

    def grab(self) -> np.ndarray:
        """Capture the window content and return an RGB uint8 array (H, W, 3)."""
        if self.hwnd:
            return self._grab_window()
        else:
            return self._grab_screen()

    def _grab_window(self) -> np.ndarray:
        """Capture using PrintWindow — works even when window is covered."""
        w, h = self.width, self.height

        # Create a device context and bitmap
        hwnd_dc = ctypes.windll.user32.GetDC(self.hwnd)
        mem_dc = ctypes.windll.gdi32.CreateCompatibleDC(hwnd_dc)
        bitmap = ctypes.windll.gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
        ctypes.windll.gdi32.SelectObject(mem_dc, bitmap)

        # PrintWindow with PW_CLIENTONLY flag (value 1) to get client area only
        ctypes.windll.user32.PrintWindow(self.hwnd, mem_dc, 1)

        # Read bitmap data
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.c_uint32),
                ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16),
                ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32),
                ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32),
                ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h  # negative = top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0  # BI_RGB

        buf = (ctypes.c_char * (w * h * 4))()
        ctypes.windll.gdi32.GetDIBits(
            mem_dc, bitmap, 0, h, buf, ctypes.byref(bmi), 0
        )

        # Cleanup
        ctypes.windll.gdi32.DeleteObject(bitmap)
        ctypes.windll.gdi32.DeleteDC(mem_dc)
        ctypes.windll.user32.ReleaseDC(self.hwnd, hwnd_dc)

        # Convert BGRA buffer to RGB numpy array
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        frame = frame[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
        return frame

    def _grab_screen(self) -> np.ndarray:
        """Fallback: capture screen region using mss."""
        import mss
        left, top, width, height = self._region
        with mss.mss() as sct:
            shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
            frame = np.array(shot, dtype=np.uint8)
            frame = frame[:, :, :3][:, :, ::-1].copy()
            return frame
