import numpy as np
import imageio


class VideoRecorder:
    def __init__(self, fps: int = 10):
        self.fps = fps
        self.frames: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray):
        self.frames.append(frame)

    def save(self, path: str):
        if not self.frames:
            return
        writer = imageio.get_writer(path, fps=self.fps)
        for frame in self.frames:
            writer.append_data(frame)
        writer.close()

    def reset(self):
        self.frames.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __len__(self):
        return len(self.frames)
