from IPython.display import Video
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from EnvironmentProperties import CellType
from EnvironmentProperties import Color 
from EnvironmentProperties import DoorState                   
import numpy as np
import os

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
        self.writer = imageio.get_writer( self.path, fps=self.fps, macro_block_size=1,  quality=8 )
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


"""
EpisodeRecorder — sprite-based version.
Tiles are extracted at runtime from the reference image supplied at construction.
No external sprite sheet required; works for any board dimensions and object config.

Usage
-----
recorder = EpisodeRecorderWithSprites(
    path        = "videos/run.mp4",
    env         = env,
    sprite_path = "assets/proyect_agent_env_image.png",
    fps         = 4,
    cell_px     = 80,
)
env.reset()
recorder.start()
while not done:
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    step_n       += 1
    recorder.capture(step_n, reward, total_reward, done)
recorder.save()
"""

import os
import numpy as np
from PIL import Image, ImageDraw


class EpisodeRecorderWithSprites:
    """
    Records an RL episode as an MP4 video using sprite tiles extracted
    from a reference dungeon image.  The three-method API (start / capture / save)
    plugs directly into any external training or evaluation loop.

    Parameters
    ----------
    path        : output MP4 file path
    env         : TwoRoomMDP instance (read-only during recording)
    sprite_path : path to the reference dungeon image
    fps         : frames per second
    cell_px     : pixel size of each grid cell in the output video
    """

    # ── Sprite crop regions (x1, y1, x2, y2) in the 2172×724 reference image ──
    _CROP_REGIONS = {
        "floor":        ( 600,  480,  900,  700),
        "floor_right":  (1300,  500, 1700,  700),
        "wall_top":     ( 500,    2,  800,   95),
        "wall_side":    (   3,  300,   90,  550),
        "agent":        ( 455,  295,  600,  478),
        "key":          ( 265,  200,  415,  312),
        "ball":         ( 735,  295,  935,  468),
        "door_locked":  (1018,  345, 1155,  510),
        "door_wall":    (1025,  155, 1155,  340),
        "door_open":    (1025,  155, 1155,  340),   # same column, tinted green in _tint()
        "door_unblocked":(1018, 345, 1155,  510),   # same as locked, tinted orange
        "exit":         (1775,   95, 2065,  445),
    }

    # ── sprites.png  (1536 × 1024) ───────────────────────────────────────────────
    # Coordinates verified pixel-by-pixel via brightness scan.
    # Background is pure black so any black padding disappears on the floor tile.
    
    _CROP_REGIONS = {
        "floor":          ( 222,  758,  436,  988),   # floor_mid tile
        "floor_right":    (  12,  758,  210,  988),   # floor_sandy tile
        "wall_top":       ( 242,  296,  652,  488),   # sandy brick — top/bottom border
        "wall_side":      (   8,  292,  222,  488),   # ivy stone   — left/right border
        "agent":          ( 148,  540,  302,  718),   # robot agent
        "key":            ( 330,  546,  462,  664),   # golden key
        "ball":           ( 476,  516,  650,  714),   # red ball
        "door_locked":    ( 791,  510,  926,  714),   # blue safe locked
        "door_wall":      ( 668,  294, 1058,  488),   # dark stone wall — divider column
        "door_open":      (1341,  510, 1490,  714),   # green glowing open
        "door_unblocked": ( 969,  510, 1117,  714),   # blue safe unlocked
        "exit":           ( 668,    8, 1528,  272),   # glowing arch — GOAL
        # ── new keys from sprites.png ────────────────────────────────────────
        "arch_locked":    ( 680,  742,  886, 1008),   # stone arch + blue door
        "arch_unblocked": ( 896,  742, 1104, 1008),   # stone arch + orange glow
        "arch_open":      (1114,  742, 1332, 1008),   # stone arch + green open
        "exit_closed":    (  28,   10,  638,  272),   # dark arch — START background
    }

    # HUD colours (RGB)
    _HUD_BG      = (30,  30,  30)
    _HUD_TEXT    = (240, 240, 240)
    _REWARD_POS  = (60,  200,  60)
    _REWARD_NEG  = (220,  60,  60)

    def __init__(
        self,
        path:        str,
        env,
        sprite_path: str,
        fps:         int = 4,
        cell_px:     int = 80,
    ):
        try:
            import imageio          # noqa: F401  (checked early, used in save())
        except ImportError:
            raise ImportError("Install with: pip install imageio imageio-ffmpeg")

        self.path    = path
        self.env     = env
        self.fps     = fps
        self.cell_px = cell_px
        self.frames  = []

        self._sprites = self._load_sprites(sprite_path, cell_px)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Call once immediately after env.reset() — captures the initial frame."""
        self.frames = [self._make_frame(step_n=0, reward=0.0, total_reward=0.0, done=False)]

    def capture(self, step_n: int, reward: float, total_reward: float, done: bool) -> None:
        """Call once per step, after env.step(), to capture that frame."""
        self.frames.append(self._make_frame(step_n, reward, total_reward, done))

    def save(self) -> str:
        """
        Writes all captured frames to MP4 and returns the absolute path.
        Call once after the episode loop ends.
        """
        import imageio

        if not self.frames:
            raise RuntimeError("No frames to save. Did you call start() and capture()?")

        # Hold final frame for 2 extra seconds so the viewer can read it
        for _ in range(self.fps * 2):
            self.frames.append(self.frames[-1]) 

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

        # Convert PIL Images → numpy arrays for imageio
        np_frames = [np.array(f) for f in self.frames]
        imageio.mimwrite(self.path, np_frames, fps=self.fps, macro_block_size=1, quality=8)

        abs_path = os.path.abspath(self.path)
        print(f"[EpisodeRecorder] {len(self.frames)} frames → {abs_path}")
        self.frames.clear()   # ← free after write
        return abs_path

    # ─────────────────────────────────────────────────────────────────────────
    # Sprite loading
    # ─────────────────────────────────────────────────────────────────────────

    
    def _load_sprites(self, sprite_path: str, cell_px: int) -> dict:
        """
        Crops each named region from the reference image and resizes to cell_px².
        Also synthesises door_open and door_unblocked by colour-tinting the base crops.
        """
        ref = Image.open(sprite_path).convert("RGB")
        size = (cell_px, cell_px)
        sprites = {}
        for name, region in self._CROP_REGIONS.items():
            sprites[name] = ref.crop(region).resize(size, Image.LANCZOS)

        # Tint door states so they're visually distinct
        sprites["door_unblocked"] = self._tint(sprites["door_unblocked"], (255, 160, 60),  alpha=0.35)
        sprites["door_open"]      = self._tint(sprites["door_open"],      (60,  220, 100), alpha=0.45)

        return sprites

    @staticmethod
    def _tint(image: Image.Image, rgb: tuple, alpha: float) -> Image.Image:
        """Overlay a solid colour at `alpha` opacity onto `image`."""
        overlay = Image.new("RGB", image.size, rgb)
        return Image.blend(image, overlay, alpha)

    # ─────────────────────────────────────────────────────────────────────────
    # Frame rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _make_frame(self, step_n: int, reward: float,
                    total_reward: float, done: bool) -> Image.Image:
        env     = self.env
        S       = self._sprites
        px      = self.cell_px
        nrows   = env.nrows
        ncols   = env.ncols
        HUD_H   = 60
        W       = ncols * px
        H       = nrows * px

        canvas = Image.new("RGB", (W, H + HUD_H))
        draw   = ImageDraw.Draw(canvas)

        # ── Identify structural positions ─────────────────────────────────
        door_color = env.door_color
        door       = env.doors[door_color]
        door_pos   = door["pos"]       # (row, col)
        door_state = door["door_state"]
        door_col   = door_pos[1]
        goal_pos   = env.end_state_pos
        start_pos  = env.initial_state

        # Object positions from env
        key_positions  = set()
        ball_positions = set()
        for pos, items in env.objects.items():
            for item in items:
                if item["type"] == CellType.KEY:
                    key_positions.add(pos)
                elif item["type"] == CellType.BALL:
                    ball_positions.add(pos)

        # ── Layer 1: floor + walls ────────────────────────────────────────
        for i in range(nrows):
            for j in range(ncols):
                x, y = j * px, i * px
                pos  = (i, j)

                # Wall cell
                if env.grid[i][j] == CellType.WALL.code:
                    if i == 0 or i == nrows - 1:
                        tile = S["wall_top"]
                    else:
                        tile = S["wall_side"]
                    canvas.paste(tile, (x, y))
                    continue

                # Divider column (door column walls above/below the door)
                if j == door_col and pos != door_pos:
                    canvas.paste(S["door_wall"], (x, y))
                    continue

                # Floor: right room has a warmer tone
                floor_tile = S["floor_right"] if j > door_col else S["floor"]
                canvas.paste(floor_tile, (x, y))

        # ── Layer 2: door ─────────────────────────────────────────────────
        dr, dc = door_pos
        dx, dy = dc * px, dr * px
        door_sprite_key = {
            DoorState.LOCKED:    "door_locked",
            DoorState.UNBLOCKED: "door_unblocked",
            DoorState.OPEN:      "door_open",
        }.get(door_state, "door_locked")
        canvas.paste(S[door_sprite_key], (dx, dy))

        # ── Layer 3: goal (exit arch) ─────────────────────────────────────
        gr, gc = goal_pos
        canvas.paste(S["exit"], (gc * px, gr * px))

        # ── Layer 4: items (key / ball) ───────────────────────────────────
        for pos in key_positions:
            canvas.paste(S["key"], (pos[1] * px, pos[0] * px))
        for pos in ball_positions:
            canvas.paste(S["ball"], (pos[1] * px, pos[0] * px))

        # ── Layer 5: agent ────────────────────────────────────────────────
        ai, aj = env.state
        canvas.paste(S["agent"], (aj * px, ai * px))

        # ── Layer 6: start label (S) ──────────────────────────────────────
        si, sj = start_pos
        draw.text((sj * px + 4, si * px + 4), "S", fill=(100, 180, 255))

        # ── Layer 7: reward flash (top-right corner) ──────────────────────
        if reward != 0.0 and not done:
            sign  = "+" if reward > 0 else ""
            color = self._REWARD_POS if reward > 0 else self._REWARD_NEG
            draw.text((W - 70, 6), f"{sign}{reward:.2f}", fill=color)

        # ── Layer 8: HUD strip ────────────────────────────────────────────
        draw.rectangle([(0, H), (W, H + HUD_H)], fill=self._HUD_BG)
        inv_str = "".join(
            f"[{it['type'].name[0]}{it['color'].name[0]}]"
            for it in env.agent_inventory
        ) or "empty"
        door_label = door_state.name[:4]
        draw.text((8, H +  6), f"Step:{step_n:3d}  Rwd:{reward:+.2f}  Total:{total_reward:+.2f}",
                  fill=self._HUD_TEXT)
        draw.text((8, H + 30), f"Inv:{inv_str}  Door:{door_label}  Cells:{len(env.visited_cells)}",
                  fill=self._HUD_TEXT)

        # ── Layer 9: DONE overlay ─────────────────────────────────────────
        if done:
            overlay = Image.new("RGBA", (W, H + HUD_H), (0, 0, 0, 120))
            canvas  = canvas.convert("RGBA")
            canvas  = Image.alpha_composite(canvas, overlay).convert("RGB")
            draw    = ImageDraw.Draw(canvas)
            draw.text((W // 2 - 28, (H + HUD_H) // 2 - 10), "DONE",
                      fill=(60, 220, 120))

        return canvas