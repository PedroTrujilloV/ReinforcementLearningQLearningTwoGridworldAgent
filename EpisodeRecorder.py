from IPython.display import Video
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from EnvironmentProperties import CellType
from EnvironmentProperties import Color 
from EnvironmentProperties import DoorState                   
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", message=".*probesize.*")


try:
    import cv2  
    import imageio
    import imageio_ffmpeg
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "imageio", "imageio-ffmpeg", "opencv-python"], check=True)
    #!pip install opencv-python #in case subprocess doesn't work.
    #!pip install imageio imageio-ffmpeg #in case subprocess doesn't work.
    import imageio
    
class EpisodeRecorder:
    """
    Records an episode as an MP4 video by capturing frames during
    an external training or run loop.

    Usage
    -----
    recorder = EpisodeRecorder("videos/run.mp4", env, fps=4, cell_px=80)
    env.reset()
    recorder.start()
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        recorder.capture(reward, total_reward, done)
        state = next_state
    recorder.save()
    """

    def __init__(self, path: str, env, fps: int = 4, cell_px: int = 80, padding: int = 10, verbose = True):
        try:
            import imageio
        except ImportError:
            raise ImportError("Install with: pip install imageio imageio-ffmpeg")

        self._cv2 = cv2
        self._np  = np

        self.path    = path
        self.env     = env
        self.fps     = fps
        self.cell_px = cell_px
        self.padding = padding
        self.frames  = []
        self.verbose = verbose

        # ── Palette (BGR) ─────────────────────────────────────────────────
        self.C = {
            "bg":             (235, 240, 245),
            "grid_line":      (190, 195, 200),
            "wall":           ( 50,  50,  50),
            "empty":          (235, 240, 245),
            "agent":          (230, 160,  50),
            "goal":           (100, 200,  60),
            "key":            ( 30, 210, 220),
            "ball":           ( 60,  80, 220),
            "door_locked":    (  0,  60, 200),
            "door_unblocked": ( 20, 160, 230),
            "door_open":      (100, 200,  60),
            "hud_bg":         ( 30,  30,  30),
            "hud_text":       (240, 240, 240),
            "reward_pos":     ( 60, 200,  60),
            "reward_neg":     ( 60,  60, 220),
        }

    def start2(self) -> None:
        """Call once after env.reset() — captures the initial frame."""
        self.frames = [self._make_frame(step_n=0, reward=0.0, total_reward=0.0, done=False)]

    def start(self) -> None:
        """Call once after env.reset() — captures the initial frame."""
        self.writer = imageio.get_writer( self.path, fps=self.fps, macro_block_size=1,  quality=8, 
                                          format="ffmpeg", codec="libx264", ffmpeg_params=["-r", str(self.fps)])
        frame = self._make_frame(step_n=0, reward=0.0, total_reward=0.0, done=False)
        self.writer.append_data(frame)

    def capture2(self, step_n: int, reward: float, total_reward: float, done: bool) -> None:
        """Call once per step after env.step() to capture that frame."""
        self.frames.append(self._make_frame(step_n, reward, total_reward, done))

    def capture(self, step_n: int, reward: float, total_reward: float, done: bool) -> None:
        """Call once per step after env.step() to capture that frame."""
        frame = self._make_frame(step_n, reward, total_reward, done)
        self.writer.append_data(frame)


    def save(self) -> str:
        """
        Writes all captured frames to MP4 and returns the absolute path.
        Call once after the episode loop ends.
        """
        final_frame = self._make_frame( step_n=-1, reward=0.0, total_reward=0.0, done=True )
        for _ in range(self.fps * 2):
            self.writer.append_data(final_frame)
        self.writer.close()
        if self.verbose:
            print(f"Saved to {self.path}")
        return os.path.abspath(self.path)

    def save2(self) -> str:
        """
        Writes all captured frames to MP4 and returns the absolute path.
        Call once after the episode loop ends.
        """

        if not self.frames:
            raise RuntimeError("No frames captured. Did you call start() and capture()?")

        # Hold final frame for 2 s
        for _ in range(self.fps * 2):
            self.frames.append(self.frames[-1])

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        imageio.mimwrite(self.path, self.frames, fps=self.fps, macro_block_size=1, quality=8)

        abs_path = os.path.abspath(self.path)
        if self.verbose:
            print(f"[EpisodeRecorder] {len(self.frames)} frames → {abs_path}")
        self.frames.clear()   # <- free memory immediately after write
        return abs_path
        

    def _make_frame(self, step_n: int, reward: float, total_reward: float, done: bool):
        cv2     = self._cv2           # <- local alias, no repeated import
        np      = self._np
        env     = self.env
        C       = self.C
        cell_px = self.cell_px
        padding = self.padding
        FONT    = cv2.FONT_HERSHEY_SIMPLEX
        HUD_H   = 56
        nrows, ncols = env.nrows, env.ncols
        W = ncols * cell_px + 2 * padding
        H = nrows * cell_px + 2 * padding + HUD_H
        img = np.full((H, W, 3), C["bg"], dtype=np.uint8)

        def cell_rect(i, j):
            x = padding + j * cell_px
            y = padding + i * cell_px
            return x, y, x + cell_px, y + cell_px

        for i in range(nrows):
            for j in range(ncols):
                x, y, x2, y2 = cell_rect(i, j)
                pos = (i, j)

                if env.grid[i][j] == CellType.WALL.code:
                    cv2.rectangle(img, (x, y), (x2, y2), C["wall"], -1)
                    continue

                cv2.rectangle(img, (x, y), (x2, y2), C["empty"], -1)

                if pos == env.end_state_pos:
                    m = cell_px // 6
                    cv2.rectangle(img, (x+m, y+m), (x2-m, y2-m), C["goal"], -1)
                    cv2.putText(img, "G", (x+cell_px//2-8, y+cell_px//2+8),
                                FONT, 0.65, (255,255,255), 2, cv2.LINE_AA)

                for _, door in env.doors.items():
                    if door["pos"] == pos:
                        ds = door["door_state"]
                        dc = (C["door_locked"]    if ds == DoorState.LOCKED
                              else C["door_unblocked"] if ds == DoorState.UNBLOCKED
                              else C["door_open"])
                        cv2.rectangle(img, (x+2, y+2), (x2-2, y2-2), dc, -1)
                        label = {DoorState.LOCKED:"D", DoorState.UNBLOCKED:"U", DoorState.OPEN:"O"}.get(ds,"?")
                        cv2.putText(img, label, (x+cell_px//2-8, y+cell_px//2+8),
                                    FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)

                for item in env.objects.get(pos, []):
                    cx_, cy_ = x+cell_px//2, y+cell_px//2
                    if item["type"] == CellType.KEY:
                        cv2.circle(img, (cx_, cy_), cell_px//5, C["key"], -1)
                        cv2.putText(img, "K", (cx_-7, cy_+6), FONT, 0.45, (20,20,20), 1, cv2.LINE_AA)
                    elif item["type"] == CellType.BALL:
                        cv2.circle(img, (cx_, cy_), cell_px//4, C["ball"], -1)
                        cv2.putText(img, "B", (cx_-7, cy_+6), FONT, 0.45, (255,255,255), 1, cv2.LINE_AA)

                cv2.rectangle(img, (x, y), (x2, y2), C["grid_line"], 1)

        ai, aj = env.state
        ax, ay, ax2, ay2 = cell_rect(ai, aj)
        cx_, cy_ = (ax+ax2)//2, (ay+ay2)//2
        r = cell_px // 3
        cv2.circle(img, (cx_, cy_), r, C["agent"], -1)
        cv2.circle(img, (cx_, cy_), r, (20, 100, 180), 2)
        cv2.putText(img, "A", (cx_-8, cy_+8), FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)

        if reward != 0.0 and not done:
            col  = C["reward_pos"] if reward > 0 else C["reward_neg"]
            sign = "+" if reward > 0 else ""
            cv2.putText(img, f"{sign}{reward:.2f}", (W-90, padding+22),
                        FONT, 0.65, col, 2, cv2.LINE_AA)

        hud_y = H - HUD_H
        cv2.rectangle(img, (0, hud_y), (W, H), C["hud_bg"], -1)
        inv_str = "".join(f"[{it['type'].name[0]}{it['color'].name[0]}]"
                          for it in env.agent_inventory) or "empty"
        door_info = next(iter(env.doors.values()))
        lines = [
            f"Step:{step_n:3d}  Rwd:{reward:+.2f}  Total:{total_reward:+.2f}",
            f"Inv:{inv_str}  Door:{door_info['door_state'].name[:4]}  Cells:{len(env.visited_cells)}",
        ]
        for li, line in enumerate(lines):
            cv2.putText(img, line, (padding, hud_y+18+li*20),
                        FONT, 0.42, C["hud_text"], 1, cv2.LINE_AA)

        if done:
            overlay = img.copy()
            cv2.rectangle(overlay, (0,0), (W,H), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.putText(img, "DONE", (W//2-50, H//2),
                        FONT, 1.4, (60,220,120), 3, cv2.LINE_AA)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)





####################

import json


class SpriteEpisodeRecorder(EpisodeRecorder):
    """
    Renders environment frames using pixel-art sprites instead of plain
    OpenCV geometric shapes.  All recording mechanics (start / capture /
    save) are inherited unchanged from EpisodeRecorder.

    Parameters
    ----------
    path : str
        Output MP4 path.
    env : object
        The grid environment.
    sprite_sheet_path : str
        Filesystem path to the sprite sheet PNG (BGRA or convertible).
    sprite_json : str | dict
        Path to the JSON produced by the sprite packer, or the already-
        parsed dict.
    fps : int
        Frames per second (default 4).
    cell_px : int
        Pixel size of one grid cell.  Sprites are NOT forced to this size;
        it only drives layout geometry.  Defaults to 99.
    padding : int
        Pixel padding around the grid canvas (default 10).
    verbose : bool
        Print save path when done (default True).
    """

    def __init__(
        self,
        path: str,
        env,
        sprite_sheet_path: str,
        sprite_json,            # str (file path) OR dict
        fps: int = 4,
        cell_px: int = 99,
        padding: int = 10,
        verbose: bool = True,
    ):
        super().__init__(path, env, fps=fps, cell_px=cell_px,
                         padding=padding, verbose=verbose)

        # ── Load JSON ─────────────────────────────────────────────────────
        if isinstance(sprite_json, str):
            with open(sprite_json, "r") as fh:
                sprite_json = json.load(fh)

        # ── Load sprite sheet as BGRA ──────────────────────────────────────
        sheet = cv2.imread(sprite_sheet_path, cv2.IMREAD_UNCHANGED)
        if sheet is None:
            raise FileNotFoundError(f"Sprite sheet not found: {sprite_sheet_path}")
        if sheet.ndim == 2:
            sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGRA)
        elif sheet.shape[2] == 3:
            sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2BGRA)

        self._sprites: dict[str, np.ndarray] = {}
        self._load_sprites(sheet, sprite_json)

    # ──────────────────────────────────────────────────────────────────────
    # Sprite loading  — keeps native resolution, NO forced resize
    # ──────────────────────────────────────────────────────────────────────

    def _load_sprites(self, sheet: np.ndarray, data: dict) -> None:
        """Crop each sprite from the sheet at its native size."""
        for s in data["sprites"]:
            name = s["fileName"]
            if name.endswith(".png"):
                name = name[:-4]
            x, y, w, h = s["x"], s["y"], s["width"], s["height"]
            self._sprites[name] = sheet[y : y + h, x : x + w].copy()

    # ──────────────────────────────────────────────────────────────────────
    # Fallback tile
    # ──────────────────────────────────────────────────────────────────────

    def _missing_tile(self) -> np.ndarray:
        """Bright-magenta BGRA tile so missing sprites are obvious."""
        px   = self.cell_px
        tile = np.zeros((px, px, 4), dtype=np.uint8)
        tile[:, :, 0] = 255   # B
        tile[:, :, 2] = 255   # R  ->  magenta
        tile[:, :, 3] = 255   # fully opaque
        return tile

    # ──────────────────────────────────────────────────────────────────────
    # Sprite retrieval with flip support
    # ──────────────────────────────────────────────────────────────────────

    def _sprite(self, name: str, flip: bool = False) -> np.ndarray:
        """
        Return the named BGRA sprite (or a magenta fallback).
        When *flip* is True the image is mirrored horizontally — used for
        wall corners / edges that share a single source tile.
        """
        spr = self._sprites.get(name)
        if spr is None:
            return self._missing_tile()
        if flip:
            spr = cv2.flip(spr, 1)
        return spr

    # ──────────────────────────────────────────────────────────────────────
    # Alpha composite blit onto a 3-channel BGR canvas
    # ──────────────────────────────────────────────────────────────────────

    def _blit(self, canvas: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
        """
        Alpha-composite a BGRA *sprite* onto the BGR *canvas* at pixel (x, y).
        Oversized sprites are clipped gracefully to canvas bounds.
        """
        sh, sw = sprite.shape[:2]
        ch, cw = canvas.shape[:2]

        # Visible region in canvas space
        x0c = max(x, 0);   y0c = max(y, 0)
        x1c = min(x + sw, cw);  y1c = min(y + sh, ch)
        if x0c >= x1c or y0c >= y1c:
            return

        # Corresponding region in sprite space
        x0s = x0c - x;  y0s = y0c - y
        x1s = x0s + (x1c - x0c);  y1s = y0s + (y1c - y0c)

        src  = sprite[y0s:y1s, x0s:x1s]
        dst  = canvas[y0c:y1c, x0c:x1c].astype(np.float32)
        a    = src[:, :, 3:4].astype(np.float32) / 255.0
        bgr  = src[:, :, :3].astype(np.float32)

        canvas[y0c:y1c, x0c:x1c] = (bgr * a + dst * (1.0 - a)).astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────
    # Anchor offset  — centres oversized sprites on the cell
    # ──────────────────────────────────────────────────────────────────────

    def _anchor_offset(self, sprite_name: str, sprite: np.ndarray, flip: bool = False) -> tuple[int, int]:
        """
        Return (dx, dy) so the sprite aligns correctly to its cell.

        * Wall sprites that are taller than cell_px extend *upward*   -> negative dy.
        * Side-edge sprites that are wider than cell_px are centred   -> negative dx.
        * All other sprites (floor, objects, agent …) centre on cell  -> (0, 0).

        The sprite itself is passed in so we never key-look-up twice.
        """
        px    = self.cell_px
        h, w  = sprite.shape[:2]
        dx, dy = 0, 0

        if "top" in sprite_name: #or "corner" in sprite_name:# or "edge" in sprite_name:
            if h > px:
                dy = px - h          # shift sprite upward by the overflow

        # if "side" in sprite_name:
        #     if w > px:
        #         dx = (px - w) // 2   # centre horizontally
        if "side_edge" in sprite_name  or "top" in sprite_name or "bottom" in sprite_name:
            if w > px:
                #dx = px - w  # shift left so RIGHT edge aligns to cell boundary
                dx = 0 if flip else px - w
                

        return dx, dy

    # ──────────────────────────────────────────────────────────────────────
    # Helper: blit a named sprite with automatic anchor correction
    # ──────────────────────────────────────────────────────────────────────

    def _blit_at(self, canvas: np.ndarray, sprite_name: str, x: int, y: int,
                 flip: bool = False) -> None:
        """Retrieve sprite, compute anchor offset, then blit."""
        spr     = self._sprite(sprite_name, flip=flip)
        dx, dy  = self._anchor_offset(sprite_name, spr, flip)
        self._blit(canvas, spr, x + dx, y + dy)

    # ──────────────────────────────────────────────────────────────────────
    # Key colour mapping
    # ──────────────────────────────────────────────────────────────────────

    _KEY_COLOUR_MAP: dict[str, str] = {
        "BLUE":   "key_blue",
        "GREEN":  "key_green",
        "RED":    "key_red",
        "YELLOW": "key_yellow",
    }

    def _key_sprite_name(self, color) -> str:
        return self._KEY_COLOUR_MAP.get(color.name.upper(), "key_blue")

    # ──────────────────────────────────────────────────────────────────────
    # Wall-tile contextual selection
    # ──────────────────────────────────────────────────────────────────────

    def _wall_sprite_name_and_flip(self, env, i: int, j: int) -> tuple[str, bool]:
        """
        Choose the most contextually appropriate wall-sprite name for (i, j)
        and whether to flip it horizontally.

        Convention (row index grows downward):
            N = (i-1, j)  — upper neighbour on screen
            S = (i+1, j)  — lower neighbour on screen
            W = (i,   j-1)
            E = (i,   j+1)
        """
        nrows, ncols = env.nrows, env.ncols
        cell_type = env.grid[i][j]

        def is_floor(r: int, c: int) -> bool:
            if r < 0 or r >= nrows or c < 0 or c >= ncols:
                return False
            return True 
            # return env.grid[r][c] != CellType.WALL.code

        N = is_floor(i - 1, j)
        S = is_floor(i + 1, j)
        W = is_floor(i, j - 1)
        E = is_floor(i, j + 1)

        if S and E and N and W:
            if cell_type == CellType.WALL.code:
                return "cell_middle_wall", False
            elif cell_type == CellType.DOOR_LOCKED.code:
                return "cell_middle_wall", False
            elif cell_type == CellType.DOOR_OPEN.code:
                return "cell_middle_wall", False
            else:
                return "cell", False # Default Cell 
                
        # ── Corners (check before straights) ──────────────────────────────
        # Top-left corner of a room: open floor to south and east
        elif S and E and not N and not W:
            return "cell_top_corner", False        # native orientation

        # Top-right corner: open floor to south and west  -> mirror
        elif S and W and not N and not E:
            return "cell_top_corner", True

        # Bottom-left corner: open floor to north and east
        elif N and E and not S and not W:
            return "cell_bottom_corner", False

        # Bottom-right corner: open floor to north and west  -> mirror
        elif N and W and not S and not E:
            return "cell_bottom_corner", True

        # ── Corridors / fully enclosed ────────────────────────────────────
        elif S and W and E and not N:
            if cell_type != CellType.WALL.code:
                return "cell_top_edge", False
            else:
                return "cell_top_wall", False

        elif N and W and E and not S:
            if cell_type != CellType.WALL.code:
                return "cell_bottom_edge", False
            else:
                return "cell_bottom_wall", False
        
        elif N and S and W and not E:
            # Left side-edge: open floor is to the EAST (right) -> mirror
            return "cell_side_edge", True

        elif N and S and E and not W:
            # Right side-edge: open floor is to the WEST (left) 
            return "cell_side_edge", False

        return "cell", False # Default Cell 

       

    # ──────────────────────────────────────────────────────────────────────
    # Frame rendering  (overrides EpisodeRecorder._make_frame)
    # ──────────────────────────────────────────────────────────────────────

    def _make_frame(
        self,
        step_n: int,
        reward: float,
        total_reward: float,
        done: bool,
    ) -> np.ndarray:
        cv2_   = self._cv2
        env    = self.env
        px     = self.cell_px
        pad    = self.padding
        FONT   = cv2_.FONT_HERSHEY_SIMPLEX
        HUD_H  = 56
        nrows, ncols = env.nrows, env.ncols

        W = ncols * px + 2 * pad
        H = nrows * px + 2 * pad + HUD_H

        # H.264/yuv420p requires even dimensions
        W = (W + 1) & ~1
        H = (H + 1) & ~1

        # Black background (sprites composite against it cleanly)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        def top_left(i: int, j: int) -> tuple[int, int]:
            return pad + j * px, pad + i * px


        # ── 1. Floor and wall tiles ────────────────────────────────────────
        for i in range(nrows):
            for j in range(ncols):
                x, y = top_left(i, j)
                name, flip = self._wall_sprite_name_and_flip(env, i, j)
                self._blit_at(canvas, name, x, y, flip=flip)

        # ── 2. Goal / exit cell ────────────────────────────────────────────
        ei, ej = env.end_state_pos
        x, y = top_left(ei, ej)
        self._blit_at(canvas, "exit", x, y)

        # ── 3. Doors ──────────────────────────────────────────────────────
        for _, door in env.doors.items():
            di, dj = door["pos"]
            x, y   = top_left(di, dj)
            ds     = door["door_state"]

            # Floor base under the door so it composites cleanly
            self._blit_at(canvas, "cell_middle_wall", x, y)

            spr_name = "door_open" if ds == DoorState.OPEN else "door"
            self._blit_at(canvas, spr_name, x, y)

            # Text badge: L = Locked, U = Unblocked
            if ds != DoorState.OPEN:
                label = "L" if ds == DoorState.LOCKED else "U"
                cv2_.putText(
                    canvas, label,
                    (x + px // 2 - 7, y + px // 2 + 8),
                    FONT, 0.55, (255, 255, 255), 2, cv2_.LINE_AA,
                )

        # ── 4. Objects: keys and balls ─────────────────────────────────────
        for pos, items in env.objects.items():
            oi, oj = pos
            x, y   = top_left(oi, oj)
            for item in items:
                if item["type"] == CellType.KEY:
                    self._blit_at(canvas, self._key_sprite_name(item["color"]), x, y)
                elif item["type"] == CellType.BALL:
                    self._blit_at(canvas, "ball", x, y)

        # ── 5. Agent ──────────────────────────────────────────────────────
        ai, aj = env.state
        x, y   = top_left(ai, aj)
        self._blit_at(canvas, "agent", x, y)

        # ── 6. Step reward overlay (top-right, non-terminal steps only) ────
        if reward != 0.0 and not done:
            col  = self.C["reward_pos"] if reward > 0 else self.C["reward_neg"]
            sign = "+" if reward > 0 else ""
            cv2_.putText(
                canvas, f"{sign}{reward:.2f}",
                (W - 90, pad + 22),
                FONT, 0.65, col, 2, cv2_.LINE_AA,
            )

        # ── 7. HUD bar ────────────────────────────────────────────────────
        hud_y = H - HUD_H
        cv2_.rectangle(canvas, (0, hud_y), (W, H), self.C["hud_bg"], -1)

        inv_str = (
            "".join(
                f"[{it['type'].name[0]}{it['color'].name[0]}]"
                for it in env.agent_inventory
            )
            or "empty"
        )
        door_info = next(iter(env.doors.values()))
        lines = [
            f"Step:{step_n:3d}  Rwd:{reward:+.2f}  Total:{total_reward:+.2f}",
            f"Inv:{inv_str}  Door:{door_info['door_state'].name[:4]}  "
            f"Cells:{len(env.visited_cells)}",
        ]
        for li, line in enumerate(lines):
            cv2_.putText(
                canvas, line,
                (pad, hud_y + 18 + li * 20),
                FONT, 0.42, self.C["hud_text"], 1, cv2_.LINE_AA,
            )

        # ── 8. "DONE" overlay ─────────────────────────────────────────────
        if done:
            overlay = canvas.copy()
            cv2_.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
            cv2_.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
            cv2_.putText(
                canvas, "DONE",
                (W // 2 - 50, H // 2),
                FONT, 1.4, (60, 220, 120), 3, cv2_.LINE_AA,
            )

        # BGR -> RGB for imageio
        return cv2_.cvtColor(canvas, cv2_.COLOR_BGR2RGB)



    def debug_frame(
        self,
        step_n: int = 0,
        reward: float = 0.0,
        total_reward: float = 0.0,
        done: bool = False,
        save_path: str | None = None,
        show: bool = True,
        figsize: tuple[int, int] | None = None,
        label_sprites: bool = False,
    ) -> np.ndarray:
        """
        Render a single frame of the current environment state and display
        it inline (Jupyter / matplotlib) and/or save it to disk.
 
        Parameters
        ----------
        step_n : int
            Step number shown in the HUD (default 0).
        reward : float
            Step reward shown in the HUD (default 0.0).
        total_reward : float
            Cumulative reward shown in the HUD (default 0.0).
        done : bool
            Whether to render the "DONE" overlay (default False).
        save_path : str | None
            If given, write the frame as a PNG to this path.
        show : bool
            Display the frame via matplotlib (default True).
            Set to False when only saving is needed.
        figsize : (int, int) | None
            matplotlib figure size in inches.  Defaults to auto-scaling
            based on the canvas dimensions.
        label_sprites : bool
            When True, draw a small sprite-name label on each cell — useful
            for verifying that the correct wall variant is chosen everywhere.
 
        Returns
        -------
        np.ndarray
            The rendered RGB frame (H × W × 3, uint8).


        Usage:

        # Quickest use — show current env state inline in Jupyter
        recorder.debug_frame()
        
        # Save a PNG snapshot without displaying
        recorder.debug_frame(save_path="debug/step_42.png", show=False)
        
        # Check wall-tile assignments across the whole map
        recorder.debug_frame(label_sprites=True)
        
        # Simulate mid-episode state for a specific step
        recorder.debug_frame(step_n=17, reward=-0.1, total_reward=-1.7)
        """
        frame = self._make_frame(step_n, reward, total_reward, done)
 
        # Optional: overlay sprite-name labels for wall-tile debugging
        if label_sprites:
            frame = self._label_sprites(frame.copy())
 
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            # imageio expects RGB; cv2.imwrite expects BGR
            self._cv2.imwrite(save_path, self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))
            if self.verbose:
                print(f"[debug_frame] saved -> {os.path.abspath(save_path)}")
 
        if show:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise ImportError("matplotlib is required for show=True.  "
                                  "Install with: pip install matplotlib")
 
            h, w = frame.shape[:2]
            if figsize is None:
                # ~96 dpi equivalent so the image isn't microscopic or huge
                figsize = (max(4, w / 96), max(3, h / 96))
 
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(frame)
            ax.axis("off")
            ax.set_title(
                f"step={step_n}  reward={reward:+.2f}  total={total_reward:+.2f}",
                fontsize=9, pad=4,
            )
            plt.tight_layout(pad=0.5)
            plt.show()
 
        return frame
 
    def _label_sprites(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay each cell with a tiny white label showing which sprite name
        was chosen.  Only used when debug_frame(label_sprites=True).
        """
        cv2_ = self._cv2
        env  = self.env
        px   = self.cell_px
        pad  = self.padding
        FONT = cv2_.FONT_HERSHEY_SIMPLEX
 
        # Convert RGB -> BGR for putText, then back
        bgr = cv2_.cvtColor(frame, cv2_.COLOR_RGB2BGR)
 
        for i in range(env.nrows):
            for j in range(env.ncols):
                x = pad + j * px
                y = pad + i * px

                name, _ = self._wall_sprite_name_and_flip(env, i, j)
 
                # Abbreviate long names so they fit inside a cell
                short = (name
                         .replace("cell_", "")
                         .replace("_wall", "_w")
                         .replace("_corner", "_c")
                         .replace("_edge", "_e")
                         .replace("middle", "mid"))
 
                # Dark shadow then white text for readability on any tile
                org = (x + 3, y + px - 6)
                cv2_.putText(bgr, short, org, FONT, 0.28, (0, 0, 0),   2, cv2_.LINE_AA)
                cv2_.putText(bgr, short, org, FONT, 0.28, (255,255,255), 1, cv2_.LINE_AA)
 
        return cv2_.cvtColor(bgr, cv2_.COLOR_BGR2RGB)