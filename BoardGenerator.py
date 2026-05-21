import random
import numpy as np
import copy
from copy import deepcopy
class BoardGenerator:
    """
    Generates randomized valid boards for TwoRoomMDP.
    board_4x9 = [
                [' ',      ' ',  ['KEY_blue','KEY_red'],          ' ',  '#',                ' ', ' ', ' ', 'END'  ],
                ['S',      ' ',             'KEY_green',          ' ',  '#',                ' ', ' ', ' ',   ' '  ],
                [' ',      ' ',                     ' ',          ' ',  '#',                ' ', ' ', ' ',   ' '  ],
                [' ',      ' ',                     ' ',   'BALL_red',  'DOOR_LOCKED_blue', ' ', ' ', ' ',   ' '  ]
            ]

    Usage
    -----
    gen = BoardGenerator(board_copy)
    board = gen.generate()

    # New dimensions
    board = gen.generate(dimensions=(7, 15))

    # Fix door/ball position, randomize everything else
    board = gen.generate(random_door_ball=False)
    """

    def __init__(self, board_copy: list[list]):
        self.board_copy = copy.deepcopy(board_copy)
        self._validate_board(self.board_copy)

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def generate(
        self,
        dimensions:       tuple[int, int] | None = None,
        random_door_ball: bool = True,
        random_keys:      bool = True,
        random_exit:      bool = True,
    ) -> list[list]:
        """
        Generates a new randomized valid board.

        Parameters
        ----------
        dimensions       : (nrows, ncols) or None to keep original size.
                           ncols must be > divider_col + 1.
        random_door_ball : randomize door row and ball position next to it.
        random_keys      : randomize key positions in left room.
        random_exit      : randomize exit position in right room.

        Returns
        -------
                list[list] : valid board ready to pass to TwoRoomMDP."""
        
        
        nrows, ncols, divider_col = self._resolve_dimensions(dimensions)
        self._validate_dimensions(nrows, ncols, divider_col)
    
        board = self._empty_board(nrows, ncols)
        self._build_divider(board, nrows, divider_col)
    
        door_row = self._place_door(board, nrows, divider_col, random_door_ball)
        self._place_balls(board, nrows, ncols, divider_col, door_row, random_door_ball)  # ← first
        self._place_start(board, divider_col)
        self._place_keys(board, divider_col, random_keys, door_row)
        self._place_exits(board, ncols, divider_col, random_exit)
    
        return board
        
        

    # ─────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────

    def _validate_board(self, board: list[list]) -> None:
        """Ensures the template board has exactly one door and one exit."""
        doors = self._extract_objects(board, prefix="DOOR_")
        exits = self._extract_objects(board, prefix="END")
        keys  = self._extract_objects(board, prefix="KEY_")

        if len(doors) == 0:
            raise ValueError("BoardGenerator: board_copy must contain exactly one DOOR_.")
        if len(exits) == 0:
            raise ValueError("BoardGenerator: board_copy must contain at least one END.")
        if len(keys) == 0:
            raise ValueError("BoardGenerator: board_copy must contain at least one KEY_.")

    def _validate_dimensions(self, nrows: int, ncols: int, divider_col: int) -> None:
        if nrows < 3:
            raise ValueError(f"BoardGenerator: nrows must be >= 3, got {nrows}.")
        if divider_col <= 0 or divider_col >= ncols - 1:
            raise ValueError(
                f"BoardGenerator: divider_col={divider_col} leaves no room on one side "
                f"for ncols={ncols}."
            )
        if (ncols - divider_col - 1) < 1:
            raise ValueError("BoardGenerator: right room must have at least 1 column.")

    # ─────────────────────────────────────────────────────────────
    # Dimension resolution
    # ─────────────────────────────────────────────────────────────

    def _resolve_dimensions(self, dimensions) -> tuple[int, int, int]:
        """Returns (nrows, ncols, divider_col) from dimensions or board_copy."""
        original_nrows = len(self.board_copy)
        original_ncols = len(self.board_copy[0])
        divider_col    = self._find_divider_col(self.board_copy)

        if dimensions is None:
            return original_nrows, original_ncols, divider_col

        nrows, ncols = dimensions
        # Keep divider proportional when dimensions change
        ratio       = divider_col / original_ncols
        new_divider = max(1, min(int(round(ncols * ratio)), ncols - 2))
        return nrows, ncols, new_divider

    def _find_divider_col(self, board: list[list]) -> int:
        """Finds the divider column by locating the DOOR_ cell."""
        for i in range(len(board)):
            for j in range(len(board[0])):
                cell  = board[i][j]
                items = cell if isinstance(cell, list) else [cell]
                if any(isinstance(it, str) and it.startswith("DOOR_") for it in items):
                    return j
        raise ValueError("BoardGenerator: no DOOR_ found in board_copy.")

    # ─────────────────────────────────────────────────────────────
    # Board construction helpers
    # ─────────────────────────────────────────────────────────────

    def _empty_board(self, nrows: int, ncols: int) -> list[list]:
        return [[" " for _ in range(ncols)] for _ in range(nrows)]

    def _build_divider(self, board: list[list], nrows: int, divider_col: int) -> None:
        for i in range(nrows):
            board[i][divider_col] = "#"

    def _place_door(self, board, nrows, divider_col, randomize) -> int:
        original_door_row = self._find_divider_col.__func__  # we'll find it below
        # Find original door row
        orig_door_row = 0
        for i in range(len(self.board_copy)):
            cell  = self.board_copy[i][divider_col] if divider_col < len(self.board_copy[0]) else " "
            items = cell if isinstance(cell, list) else [cell]
            if any(isinstance(it, str) and it.startswith("DOOR_") for it in items):
                orig_door_row = i
                break

        door_value = self._find_door_value(self.board_copy)
        door_row   = random.randint(0, nrows - 1) if randomize else min(orig_door_row, nrows - 1)
        board[door_row][divider_col] = door_value
        return door_row

    def _place_start(self, board, divider_col) -> None:
        pos = self._random_empty(board, side="left", divider_col=divider_col)
        self._place(board, "S", pos)

    def _place_keys2(self, board, divider_col, randomize) -> None:
        keys = self._extract_objects(self.board_copy, prefix="KEY_")
        for key in keys:
            # Keys always go in the left room (game rule)
            pos = self._random_empty(board, side="left", divider_col=divider_col)
            self._place(board, key, pos)

    def _place_keys(self, board, divider_col, randomize, door_row) -> None:
        keys = self._extract_objects(self.board_copy, prefix="KEY_")
        for key in keys:
            pos = self._random_empty_or_stackable(board, side="left", divider_col=divider_col, door_row= door_row)
            self._place(board, key, pos)

    def _place_balls(self, board, nrows, ncols, divider_col, door_row, randomize) -> None:
        balls = self._extract_objects(self.board_copy, prefix="BALL_")
        for ball in balls:
            candidate = (door_row, divider_col - 1)
            if board[candidate[0]][candidate[1]] == " ":
                pos = candidate
            else:
                raise RuntimeError(
                    f"BoardGenerator: cell {candidate} immediately left of the door "
                    f"is occupied. Cannot place ball. Check object placement logic."
                )
            self._place(board, ball, pos)

    def _place_exits(self, board, ncols, divider_col, randomize) -> None:
        exits = self._extract_objects(self.board_copy, prefix="END")
        for exit_obj in exits:
            # Exits always go in the right room (game rule)
            pos = self._random_empty(board, side="right", divider_col=divider_col)
            self._place(board, exit_obj, pos)

    # ─────────────────────────────────────────────────────────────
    # Placement primitives
    # ─────────────────────────────────────────────────────────────

    def _random_empty(self, board, side: str, divider_col: int) -> tuple[int, int]:
        """
        Returns a random empty cell on the requested side.
        side: "left" | "right"
        Raises RuntimeError if no empty cell is found after max attempts.
        """
        nrows = len(board)
        ncols = len(board[0])
        max_attempts = nrows * ncols * 10

        for _ in range(max_attempts):
            i = random.randint(0, nrows - 1)
            j = (
                random.randint(0, divider_col - 1)      if side == "left"
                else random.randint(divider_col + 1, ncols - 1)
            )
            if board[i][j] == " ":
                return (i, j)

        raise RuntimeError(
            f"BoardGenerator: no empty cell found on side='{side}' after {max_attempts} attempts. "
            f"Board may be too small for the number of objects."
        )

    def _place(self, board, obj, pos) -> None:
        i, j = pos
        if board[i][j] == " ":
            board[i][j] = obj
        elif isinstance(board[i][j], list):
            board[i][j].append(obj)
        else:
            board[i][j] = [board[i][j], obj]

    # ─────────────────────────────────────────────────────────────
    # Object extraction from template
    # ─────────────────────────────────────────────────────────────

    def _extract_objects(self, board, prefix: str) -> list[str]:
        found = []
        for row in board:
            for cell in row:
                items = cell if isinstance(cell, list) else [cell]
                for item in items:
                    if isinstance(item, str) and item.startswith(prefix):
                        found.append(item)
        return found

    def _find_door_value(self, board) -> str:
        for row in board:
            for cell in row:
                items = cell if isinstance(cell, list) else [cell]
                for item in items:
                    if isinstance(item, str) and item.startswith("DOOR_"):
                        return item
        raise ValueError("BoardGenerator: no DOOR_ value found.")


    def _random_empty_or_stackable(self, board, side: str, divider_col: int, door_row: int) -> tuple[int, int]:
        """Like _random_empty but allows placing on cells that already have objects (not walls/door)."""
        nrows = len(board)
        ncols = len(board[0])
        max_attempts = nrows * ncols * 10
        ball_pos = (door_row , divider_col - 1)
        for _ in range(max_attempts):
            i = random.randint(0, nrows - 1)
            j = (
                random.randint(0, divider_col - 1)         if side == "left"
                else random.randint(divider_col + 1, ncols - 1)
            )
            cell = board[i][j]
            cell_pos = (i,j)
            # Allow empty or already-occupied (but not wall or door or ball)
            if cell == " " or (isinstance(cell, (str, list)) and cell != "#" and cell != "S" and cell_pos != ball_pos  
                               and not (isinstance(cell, str) and cell.startswith("DOOR_"))):
                return (i, j)
    
        raise RuntimeError(
            f"BoardGenerator: no stackable cell found on side='{side}' after {max_attempts} attempts."
        )