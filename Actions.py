from enum import Enum

class Action(Enum):
    """
    Enumeration of all possible agent actions in the environment.
    """

    # Movement actions
    UP = "Move Up"
    DOWN = "Move Down"
    RIGHT = "Move Right"
    LEFT = "Move Left"

    # Inventory interactions
    PICK_UP_ITEM = "Pick Up Item"
    DROP_ITEM = "Drop Item"
    USE_ITEM = "Use Item"

    # Terminal action
    EXIT = "Exit"

    @property
    def code(self):
        codes = {
            Action.UP: 0,
            Action.DOWN: 1,
            Action.RIGHT: 2,
            Action.LEFT: 3,
            Action.PICK_UP_ITEM: 4,
            Action.DROP_ITEM: 5,
            Action.USE_ITEM: 6,
            Action.EXIT: 7,
        }
        return codes[self]

    @property
    def description(self) -> str:
        descriptions = {
            Action.UP: "The agent moves one cell upward (i-1, j)",
            Action.DOWN: "The agent moves one cell downward (i+1, j)",
            Action.RIGHT: "The agent moves one cell to the right (i, j+1)",
            Action.LEFT: "The agent moves one cell to the left (i, j-1)",
            Action.PICK_UP_ITEM: (
                "Picks up the first available item in the current cell "
                "and adds it to the agent's inventory"
            ),
            Action.DROP_ITEM: (
                "Removes the first item from the agent's inventory "
                "and deposits it in the current cell"
            ),
            Action.USE_ITEM: (
                "Uses the first usable item in the agent's inventory "
                "within the current context "
                "(e.g., using a key on a locked door)"
            ),
            Action.EXIT: (
                "The agent declares that it has reached the exit; "
                "the episode ends. Only valid in the END cell"
            ),
        }
        return descriptions[self]