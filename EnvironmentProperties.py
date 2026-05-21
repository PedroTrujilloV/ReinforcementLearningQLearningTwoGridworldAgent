from enum import Enum

# -------CellType-------------------------------------------------------------------------------------

class CellType(Enum):
    """
    Enumeration of all possible cell types in the environment.
    """

    EMPTY = " "
    START = "S"
    WALL = "#"
    KEY = "KEY"
    BALL = "BALL"
    DOOR_LOCKED = "DOOR_LOCKED"
    DOOR_OPEN = "DOOR_OPEN"
    END = "END"

    @property
    def code(self):
        codes = {
            CellType.EMPTY: 0,
            CellType.START: 1,
            CellType.WALL: None,
            CellType.KEY: 5,
            CellType.BALL: 6,
            CellType.DOOR_LOCKED: 2,
            CellType.DOOR_OPEN: 3,
            CellType.END: 4,
        }
        return codes[self]

    @property
    def description(self) -> str:
        descriptions = {
            CellType.EMPTY:
                "Empty cell (walkable)",

            CellType.START:
                "Start cell",

            CellType.WALL:
                "Barrier / obstacle — not walkable",

            CellType.KEY:
                "Key cell (collectible item) — carries 'color' attribute",

            CellType.BALL:
                "Ball cell (collectible item; blocks door) — carries 'color' attribute",

            CellType.DOOR_LOCKED:
                (
                    "Locked door — carries 'color' attribute; "
                    "only a key of the same color can open it"
                ),

            CellType.DOOR_OPEN:
                "Open door (walkable)",

            CellType.END:
                "Exit cell (goal)",
        }

        return descriptions[self]


# -------Color-------------------------------------------------------------------------------------
class Color(Enum):
    """
    Enumeration of all supported environment colors.
    """

    BLUE = "blue"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    PURPLE = "purple"
    GREY = "grey"

    @property
    def description(self) -> str:
        descriptions = {
            Color.BLUE:
                "Default color for the key and locked door.",

            Color.RED:
                "Default color for the ball.",

            Color.GREEN:
                "Additional environment color option.",

            Color.YELLOW:
                "Additional environment color option.",

            Color.PURPLE:
                "Additional environment color option.",

            Color.GREY:
                "Neutral environment color option.",
        }

        return descriptions[self]

# -------DoorState-------------------------------------------------------------------------------------

class DoorState(Enum):
    """
    Enumeration of all possible door states in the environment.
    """

    LOCKED = "LOCKED"
    UNBLOCKED = "UNBLOCKED"
    OPEN = "OPEN"

    @property
    def description(self) -> str:
        descriptions = {
            DoorState.LOCKED:
                (
                    "The door is locked and cannot be opened yet. "
                    "A matching key is required."
                ),

            DoorState.UNBLOCKED:
                (
                    "The blocking object has been removed, "
                    "but the door is still closed."
                ),

            DoorState.OPEN:
                (
                    "The door has been successfully opened "
                    "and is now walkable."
                ),
        }

        return descriptions[self]
