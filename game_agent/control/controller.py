import time
from typing import Dict, List, Optional

import pydirectinput

from game_agent.actions import Action

# Default key mapping for Pong left paddle (W/S keys)
DEFAULT_KEY_MAP: Dict[Action, List[str]] = {
    Action.NOOP: [],
    Action.UP: ["w"],
    Action.DOWN: ["s"],
}


class GameController:
    def __init__(self, key_map: Optional[Dict[Action, List[str]]] = None):
        self.key_map = key_map or DEFAULT_KEY_MAP
        self._held_keys: set = set()

        # Disable pydirectinput's built-in pause between actions
        pydirectinput.PAUSE = 0.0

    def press(self, action: Action, duration: float = 0.1):
        """Press all keys for an action, hold for duration, then release."""
        self.release_all()

        keys = self.key_map.get(action, [])
        for key in keys:
            pydirectinput.keyDown(key)
            self._held_keys.add(key)

        if keys and duration > 0:
            time.sleep(duration)

        self.release_all()

    def hold(self, action: Action):
        """Press and hold keys for an action without releasing."""
        self.release_all()
        keys = self.key_map.get(action, [])
        for key in keys:
            pydirectinput.keyDown(key)
            self._held_keys.add(key)

    def release_all(self):
        """Release all currently held keys."""
        for key in self._held_keys:
            try:
                pydirectinput.keyUp(key)
            except Exception:
                pass
        self._held_keys.clear()
