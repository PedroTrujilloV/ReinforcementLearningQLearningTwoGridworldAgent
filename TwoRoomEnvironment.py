import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import copy
from copy import deepcopy
from enum import Enum
from Actions import Action
from EnvironmentProperties import CellType, Color, DoorState
from Reward import RewardEvent

import warnings
warnings.filterwarnings("ignore", message=".*probesize.*")


#### ----------------------------------------------------------------------------------------
#### --------------- Gridworld ------------------------------------------------------------
#### ----------------------------------------------------------------------------------------

class Gridworld:
    def __init__(self, board:list[list[str]], dimensions:tuple[int,int] ):
        self.nrows, self.ncols = dimensions
        self.initial_state = (0,0)
        self.state = (0,0)
        self.encode(board)

    def encode(self, board:list[list[str]]):
        """" Encode the board inside the grid"""
        self.grid = [[0 for _ in range(self.ncols)] for _ in range(self.nrows)]
        for i in range(self.nrows):
            for j in range(self.ncols):
                if board[i][j] == "#":
                    self.grid[i][j] = None
                elif board[i][j] == "S":
                    self.initial_state = (i,j)
                elif board[i][j] != " ":
                    self.grid[i][j] = int(board[i][j])
        self.state = self.initial_state
        
    def get_states(self) -> list[list[int]]:
        """Retorna todos los estados válidos (no-None) del grid como lista de (i,j)."""
        return self.grid

    def get_current_state(self) -> tuple[int, int]:
        """Construye y retorna el estado completo actual del agente."""
        return self.state
    
    def state_value(self, state:tuple[int, int]) -> float:
        return self.grid[state[0]][state[1]]
    
    def get_possible_actions(self, state: tuple) -> list[str]:
        """Retorna las acciones válidas para un estado dado"""
        actions = ['up', 'right', 'down', 'left']
        x = state[0]
        y = state [1]
        if x == 0:
            actions.remove('up')
        elif x == 3:
            actions.remove('down')
        if y == 0:
            actions.remove('left')
        elif y == 11:
            actions.remove('right')
        return actions
        
    def get_action_index(self, action: str) -> int:
        """Retorna el índice numérico de una acción dado su símbolo string."""
        actions = ['up', 'right', 'down', 'left']
        index = 0
        for a in actions:
            if action == a:
                return index
            index += 1

    def get_possible_states(self, state: tuple, action: str) -> list[tuple]:
        """Retorna los posibles estados resultantes de ejecutar action en state"""
        actions = self.get_possible_actions(state)
        action_index = self.get_action_index(action)
        probability = [0.1, 0.1, 0.8]
        rewards=[]
        states=[]
        for i in [-1, 1, 0]:
            action = actions[(action_index+i)%len(actions)]
            reward, new_state = self.simulate_action(state, action) 
            rewards.append(reward)
            states.append(new_state)
        return probability, rewards, states


    def simulate_action(self, state: tuple, action: str) -> tuple:
        """Simula un paso SIN modificar ningún atributo del ambiente (método puro)."""
        i,j = state
        if action == 'up' and i > 0 and self.grid[i-1][j] != None:
            i -= 1
        elif action == 'down' and i < self.nrows-1 and self.grid[i+1][j] != None:
            i += 1
        elif action == 'left' and j > 0 and self.grid[i][j-1] != None:
            j -= 1
        elif action == 'right' and j < self.ncols-1 and self.grid[i][j+1] != None:
            j += 1
        new_state = (i,j)
        return self.grid[i][j], new_state

        
    def do_action(self, action:str) -> tuple[float, tuple[int,int]]:
        """Ejecuta la acción y modifica el estado real del ambiente."""
        i, j = self.state
        if action == 'up' and i > 0 and self.grid[i-1][j] != None:
            i -= 1
        elif action == 'down' and i < self.nrows-1 and self.grid[i+1][j] != None:
            i += 1
        elif action == 'left' and j > 0 and self.grid[i][j-1] != None:
            j -= 1
        elif action == 'right' and j < self.ncols-1 and self.grid[i][j+1] != None:
            j += 1
        self.state = (i,j)
        return self.grid[i][j], self.state

    def reset(self):
        """Reinicia el ambiente al estado inicial."""
        self.state = self.initial_state
    
    def is_terminal(self, state:tuple[int,int]=None) -> bool:
        """Retorna True si la celda es un estaddo final"""
        if state == None:
            state = self.state
        return self.grid[state[0]][state[1]] == 1 or self.grid[state[0]][state[1]] == -1

    def render(self, mode: str = 'human') -> None:
        """Visualiza el estado actual del ambiente en consola o como imagen."""
        print("Gridworld render() Not Implemented")
        pass



ENV_MAX_LOAD   = 1   # Maximum object capacity per environment cell
AGENT_MAX_LOAD = 1   # Maximum object capacity the agent can carry; with a capacity >1, the agent will likely learn not to drop the ball
MAX_STEPS      = 500 # Step limit per episode (prevents infinite episodes)

#### ----------------------------------------------------------------------------------------
#### --------------- TwoRoomMDP ------------------------------------------------------------
#### ----------------------------------------------------------------------------------------


class TwoRoomMDP(Gridworld):
    def __init__(self, 
                 board:list[list[str]] , 
                 dimensions:tuple[int,int], 
                 max_cell_inventory: int = ENV_MAX_LOAD,
                ):
        self.max_cell_inventory = max_cell_inventory
        self.board_copy = copy.deepcopy(board)
        self.objects = {}
        self.agent_inventory = []
        self.steps = 0 
        self.doors = {}
        self.door_color = Color.BLUE
        self.visited_events = set()
        self.visited_cells = set()
        self.end_state_pos = (dimensions[0] - 1, dimensions[1] - 1)
        super().__init__( board, dimensions) 
        

    def encode(self,  board):
        """" Encode the board inside the grid and instance attributes
        using the CellType Enum, DoorState Enum, and Color Enum"""
        if not self.is_board(board):
            print("TwoRoomMDP.encode: encoding a transitions table")
        self.grid = [[CellType.EMPTY.code for _ in range(self.ncols)] for _ in range(self.nrows)]
        keys = {}
        doors = []
        doors_by_state = {}
        initial_state_pos = (0,0)
        
        for i in range(self.nrows):
            for j in range(self.ncols):
                state = (i,j)
                if not isinstance(board[i][j], list):
                    cell_list = [board[i][j]]
                else:
                    cell_list = board[i][j]
                    
                for cell_item in cell_list:
                    cell = cell_item.split("_")
                    # WALL ------------------------------------------------------------
                    if cell[0] == CellType.WALL.value:
                        self.grid[i][j] = CellType.WALL.code
                    # START ------------------------------------------------------------  
                    elif cell[0] == CellType.START.value:
                        initial_state_pos = state
                    # DOOR_LOCKED ------------------------------------------------------------ 
                    elif cell[0] == CellType.DOOR_LOCKED.name.split("_")[0]:
                        if doors_by_state.get( state,  None ) is not None:
                            raise ValueError(f"TwoRoomMDP.encode() Error: Cell must have only one DOOR ")
                        if len(cell) > 2:
                            door_color = Color(cell[2])
                        else:
                            door_color = Color.BLUE

                        if door_color in doors:
                            raise ValueError(f"TwoRoomMDP.encode() Error: Board must have only one DOOR_{door_color} ")

                        doors_by_state[state] = door_color
                        doors.append(door_color)
                        # self.has_n_valid_doors_and_keys( len(doors), len(keys))
                        door_state = DoorState(cell[1])
                        if door_state != DoorState.LOCKED:
                            raise ValueError(f"TwoRoomMDP.encode() Error: Door {door_color} must be initialized as {DoorState.LOCKED}")
                        self.doors[door_color] = {"pos": state, "door_state": door_state}
                    # END ------------------------------------------------------------ 
                    elif cell[0] == CellType.END.value:
                        self.end_state_pos = state
                        #self.grid[i][j] = 1 # We need to think about this cell encoding action                                                   
                    # KEY ------------------------------------------------------------    
                    elif cell[0] == CellType.KEY.value:
                        if len(cell) > 1:
                            key_color = Color(cell[1])
                        else:
                            key_color = Color.BLUE
                        if key_color in keys.keys():
                            raise ValueError(f"TwoRoomMDP.encode() Error:Board must have only one KEY_{key_color} ")
                        keys[key_color] = state
                        # self.has_n_valid_doors_and_keys( len(doors), len(keys))
                        cell_inventory = self.objects.get( state,  [] )
                        # if len(cell_inventory) >= self.max_cell_inventory:
                        #     print(f"TwoRoomMDP.encode() warning: cell {state} maxed the inventory capicity, skipping KEY_{key_color} ")
                        #     continue 
                        cell_inventory.insert(0, {"type": CellType.KEY, "color": key_color } ) # Make sure the key is the first object in list
                        self.objects[state] = cell_inventory
                    # BALL ------------------------------------------------------------ 
                    elif cell[0] == CellType.BALL.value:
                        if len(cell) > 1:
                            ball_color = Color(cell[1])
                        else:
                            ball_color = Color.RED
                        cell_inventory = self.objects.get( state,  [] )
                        if len(cell_inventory) >= self.max_cell_inventory:
                            print(f"TwoRoomMDP.encode() warning: cell {state} maxed the inventory capicity, skipping BALL_{ball_color}")
                            continue 
                        cell_inventory.append({"type": CellType.BALL, "color": ball_color } )
                        self.objects[state] = cell_inventory

        
        matching_colors = set(doors) & set(keys.keys())
        if len(matching_colors) == 0 :
            raise ValueError(f"""TwoRoomMDP.encode() Error: There is not a key - door color matching in the board. 
            Please ensure there is only one key and door matching colors.""")
        if len(matching_colors) > 1:
            raise ValueError(f"""TwoRoomMDP.encode() Error: There are more than 1  key - door color matching pair in the board. 
            Please ensure there is only one key and door matching colors.""")
        self.has_n_valid_doors_and_keys( len(set(doors)), len(set(keys.keys())))
        
        self.door_color = matching_colors.pop() 
        # Trim the cell inventory saflety (wtihtout erasing they door_color key)
        key_state = keys[self.door_color]
        for a_k_state in keys.values():
            cell_inventory = self.objects.get( a_k_state,  [] )
            new_cell_inventory = []
            if a_k_state == key_state:
                new_cell_inventory = [{"type": CellType.KEY, "color": self.door_color }]                
            for item in cell_inventory:
                if item["type"] == CellType.KEY and item["color"] == self.door_color:
                    continue
                if len(new_cell_inventory) >= self.max_cell_inventory:
                    break
                new_cell_inventory.append(item)
            self.objects[a_k_state] = new_cell_inventory
                
        
        #print(f"* self.door_color = {self.door_color }")
        self.initial_state = initial_state_pos
        self.state = initial_state_pos
        self.visited_cells = {initial_state_pos}

        
    def has_n_valid_doors_and_keys(self, n_doors, n_keys):
         if n_doors > 1 and n_keys > 1:
             raise ValueError("""TwoRoomMDP.encode() Error: Too many keys and doors! 
             board must have keys = 1 and doors >= 1 or viceversa but not many to many!""")
        
            
    def is_board(self, board) -> bool:
        """ Returns True if the input is a valid board of type:  list[list[str]]
        Conditions:
        - board must be a list
        - every row must be a list
        - every cell must be a string
        Valid example:
            [
                ['S', ' ', 'END'],
                ['#', 'KEY_blue', ' ']
            ]
        """
        if not isinstance(board, list):
            return False
        for row in board:
            if not isinstance(row, list):
                return False
            for cell in row:
                if not isinstance(cell, (str, list)):
                    return False
                if isinstance(cell, list):
                    for item in cell:
                        if not isinstance(item, str):
                            return False
        return True
    

    def get_current_state(self, include_snapshot = False) -> dict:
        """Obtiene el estado actual del agente usando el estado interno o state pos (i,j), 
        invoca _compute_state con ese estado actual y retorna el estado completo actual del agente (ver Sección State Variables).
        s_t = (i, j, has_key, has_ball, door_state, door_color, left_room, t) """
        return self._compute_state(
            state_pos = self.state,
            inventory = self.agent_inventory,
            doors     = self.doors,
            steps     = self.steps,
            door_color= self.door_color,
            objects   = self.objects,
            include_snapshot= include_snapshot,
        )

    def get_state(self, for_state_position: tuple[int, int], include_snapshot= False) -> dict:
        """Obtiene el estado para la  posiscion del estado solicitado en state_position"""
        return self._compute_state(
            state_pos = for_state_position,
            inventory = self.agent_inventory,
            doors     = self.doors,
            steps     = self.steps,
            door_color= self.door_color,
            objects   = self.objects,
            include_snapshot= include_snapshot,
        )

    def _compute_state( self, *, state_pos: tuple[int, int],  
                        inventory: list[dict], doors: dict, steps: int, door_color: Color, objects: dict, include_snapshot = False) -> dict:
        """ Construye y retorna el estado completo del agente.
    
        Parámetros
        ----------
        state_pos : tuple[int,int] Posición del agente (i,j)
        inventory : list[dict] Snapshot de inventario del agente a usar para computar has_key / has_ball.
        doors : dict Snapshot de puertas.
        steps : int  Número de pasos del snapshot.
        door_color : Color Color de la puerta principal.
        objects : dict Snapshot de objetos en el grid.
        
        Retorna dict:
            s_t = (i, j, has_key, has_ball, door_state, door_color, left_room, steps )
            si include_snapshot:
            + snapshot(dict, para uso exclusivo de los detectores)
        """
        assert inventory is not None
        assert doors is not None
        # --------------------------- Position ---------------------------
        i, j = state_pos
        # --------------------------- Door state ---------------------------
        door = doors[door_color]
        door_state = door["door_state"]
        door_pos = door["pos"]
        # --------------- Inventory-derived state ---------------------------
        has_key = next(( item["color"] for item in inventory if item["type"] == CellType.KEY ), None)
        has_ball = any(
            item["type"] == CellType.BALL
            for item in inventory
        )
        # Room state: False => left room, True  => right room
        left_room = j > door_pos[1]
        
    # ----------- Full observable state ---------------------------    
        observable_state = {
            "i": i,
            "j": j,
            "has_key": has_key,
            "has_ball": has_ball,
            "door_state": door_state,
            "door_color": door_color,
            "left_room": left_room,
            "steps": steps,
        }

     # --------------- create snapshot (if needed)---------------------------
        if include_snapshot:
            assert objects is not None
            snapshot = { "objects": deepcopy(objects), 
                         "inventory":  deepcopy(inventory), 
                         "doors": deepcopy(doors),
                         "visited_events": frozenset(self.visited_events),  # immutable
                       } 
            observable_state["snapshot"] = snapshot

        return observable_state
    
    
    def get_action_index(self, action: Action) -> int:
        """Retorna el índice numérico de una acción dado su símbolo string."""
        return action.code
        
    def get_possible_actions(self, state: dict) -> list[Action]:
        """
        Retorna las acciones válidas para un estado dado,
        aplicando todas las restricciones.
        """
        actions = list(Action)
        actions = self.validate_restrictions(state, actions)
        return actions
    
    
    ## ----------------------------------------------------------------------------------------
    ## --------------- ACTIONS STATE RESTRICTIONS ---------------------------------------------
    ## ----------------------------------------------------------------------------------------
    
    def validate_restrictions( self, state: dict, actions: list[Action] ) -> list[Action]:
        """ Aplica todas las restricciones del entorno y retorna únicamente las acciones válidas."""
        actions = self.validate_r_01_restrictions(state, actions)
        actions = self.validate_r_02_restrictions(state, actions)
        actions = self.validate_r_03_restrictions(state, actions)
        actions = self.validate_r_04_restrictions(state, actions)
        actions = self.validate_r_05_restrictions(state, actions)
        actions = self.validate_r_06_restrictions(state, actions)
        actions = self.validate_r_07_restrictions(state, actions)
        return actions

    
    def safe_remove_action( self, action: Action, actions: list[Action] ) -> list[Action]:
        """ Elimina una acción de forma segura. """
        if action in actions:
            actions.remove(action)
        return actions
    
    
    # R-01: GENERAL MOVEMENT RESTRICTIONS ----------------------------------------------------
    def validate_r_01_restrictions( self, state: dict, actions: list[Action] ) -> list[Action]:
        """ R-01: Restricciones generales de movimiento. """
        movement_actions = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }
        door = self.doors[self.door_color]
        door_pos = door["pos"]
        current_i = state["i"]
        current_j = state["j"]
        valid_actions = []
    
        for action in actions:
            # Non movement actions bypass R01
            if action not in movement_actions:
                valid_actions.append(action)
                continue
                
            di, dj = movement_actions[action]
            next_i = current_i + di
            next_j = current_j + dj
            next_pos = (next_i, next_j)
            
            # Outside grid
            # --------------------------------------------------------
            if (
                next_i < 0
                or next_i >= self.nrows
                or next_j < 0
                or next_j >= self.ncols
            ):
                continue
                
            # Wall cell
            # --------------------------------------------------------
            if self.grid[next_i][next_j] == CellType.WALL.code:
                continue

            # Door restrictions
            # --------------------------------------------------------
            if next_pos == door_pos:
                # Door locked/unblocked => not traversable
                if door["door_state"] != DoorState.OPEN:
                    continue
                # Door open => only horizontal traversal
                if action not in [Action.LEFT, Action.RIGHT]:
                    continue
            valid_actions.append(action)
        return valid_actions
    
    
    # R-02: PICK_UP_ITEM RESTRICTIONS ---------------------------------------------------------
    def validate_r_02_restrictions( self, state: dict, actions: list[Action] ) -> list[Action]:
        """ R-02: Restricciones para PICK_UP_ITEM. """
        state_pos = (state["i"], state["j"])
        cell_objects = state["snapshot"]["objects"].get(state_pos, [])
        ## No objects
        if len(cell_objects) == 0:
            return self.safe_remove_action( Action.PICK_UP_ITEM, actions )
        ## Inventory full
        if len(state["snapshot"]["inventory"]) >= AGENT_MAX_LOAD:
            return self.safe_remove_action( Action.PICK_UP_ITEM, actions )
        ## Allowed
        return actions
    
    
    # R-03: DROP_ITEM RESTRICTIONS ------------------------------------------------------------
    def validate_r_03_restrictions( self, state: dict, actions: list[Action] ) -> list[Action]:
        """ R-03: Restricciones para DROP_ITEM. """
        state_pos = (state["i"], state["j"])
        cell_objects = state["snapshot"]["objects"].get(state_pos, [])
        ## Empty inventory
        if len(state["snapshot"]["inventory"]) == 0:
            return self.safe_remove_action( Action.DROP_ITEM, actions )
        ## Cell full
        if len(cell_objects) >= ENV_MAX_LOAD:
            return self.safe_remove_action( Action.DROP_ITEM, actions )
        ## Cannot drop on door cell
        for _, door_data in self.doors.items():
            if state_pos == door_data["pos"]:
                return self.safe_remove_action( Action.DROP_ITEM, actions )
    
        return actions
    
    
    # R-04: USE_ITEM RESTRICTIONS -------------------------------------------------------------
    def validate_r_04_restrictions(self, state: dict, actions: list[Action] ) -> list[Action]:
        """ R-04: Restricciones para USE_ITEM. """
        door = self.doors[self.door_color]
        door_pos_i, door_pos_j = door["pos"]
        state_i = state["i"]
        state_j = state["j"]
        required_pos = (
            door_pos_i,
            door_pos_j - 1
        )
        current_pos = (state_i, state_j)

        # Must be left-adjacent to door
        # --------------------------------------------------------
        if current_pos != required_pos:
            return self.safe_remove_action( Action.USE_ITEM, actions )

        # Empty inventory
        # --------------------------------------------------------
        if len(state["snapshot"]["inventory"]) == 0:
            return self.safe_remove_action( Action.USE_ITEM, actions )
            
        # Must have correct key
        # --------------------------------------------------------
        if state["has_key"] != self.door_color:
            return self.safe_remove_action( Action.USE_ITEM, actions )

        # Door must be unblocked
        # --------------------------------------------------------
        if door["door_state"] != DoorState.UNBLOCKED:
            return self.safe_remove_action( Action.USE_ITEM, actions )
        return actions
    
    
    # R-05: EXIT RESTRICTIONS -----------------------------------------------------------------
    def validate_r_05_restrictions( self, state: dict, actions: list[Action] ) -> list[Action]:
        """ R-05: Restricciones para EXIT. """
        state_pos = (state["i"], state["j"])
        if state_pos != self.end_state_pos:
            return self.safe_remove_action( Action.EXIT, actions )
        return actions
    
    
    # R-06: GRID BOUNDARY RESTRICTIONS --------------------------------------------------------
    def validate_r_06_restrictions(self, state: dict, actions: list[Action] ) -> list[Action]:
        """ R-06: Restricciones de bordes del grid. """
        i = state["i"]
        j = state["j"]
        ## Top edge
        if i == 0:
            actions = self.safe_remove_action( Action.UP, actions )
        ## Bottom edge
        if i == self.nrows - 1:
            actions = self.safe_remove_action( Action.DOWN, actions )
        ## Left edge
        if j == 0:
            actions = self.safe_remove_action( Action.LEFT, actions )
        ## Right edge
        if j == self.ncols - 1:
            actions = self.safe_remove_action( Action.RIGHT, actions )
        return actions
    
    
    # R-07: BACKTRACKING RESTRICTION----------------------------------------------------------
    def validate_r_07_restrictions(self, state: dict, actions: list[Action] ) -> list[Action]:
        """ R-07: Backtracking está permitido. La penalización se maneja mediante R14_event. """
        return actions
        
        
    ## ----------------------------------------------------------------------------------------
    ## --------------- REWARD AND EVENT DETECTION ---------------------------------------------
    ## ----------------------------------------------------------------------------------------


    ## --------------- Movement and Exploration Rewards ---------------------------------------

    def _R1_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """Any valid transition: -0.01"""
        return RewardEvent.STEP_PENALTY
    
    
    def _R2_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """New cell visited for the first time: +0.50 """
        allowed_actions = [ Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT ]
        if action not in allowed_actions:
            return None
        next_state_pos = (next_state["i"], next_state["j"])
        if next_state_pos not in self.visited_cells:
            return RewardEvent.NEW_CELL
        return None
    
    
    def _R13_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ Penalty for remaining in / returning to the left room after opening the door: -1.00 """
        door_pos_j = state["snapshot"]["doors"][self.door_color]["pos"][1]
        if state["door_state"] != DoorState.OPEN:
            return None
        if state["left_room"] is not False:
            return None
        if state["j"] >= door_pos_j:
            return None
        if next_state["j"] < door_pos_j:
            return RewardEvent.PENALTY_LEFT_ROOM_AFTER_OPEN
        return None
    
    
    ## --------------- Ball Related Rewards ---------------------------------------
    
    def _R3_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """Arriving at a cell with a ball for the first time: +1.00"""
        allowed_actions = [ Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT ]    
        if state["has_ball"] is not False:
            return None
        if action not in allowed_actions:
            return None
        next_state_pos = (next_state["i"], next_state["j"])
        if next_state_pos in self.visited_cells:
            return None
        next_state_cell_inventory = next_state["snapshot"]["objects"].get(next_state_pos, [])
        ball = next(
            (
                item for item in next_state_cell_inventory
                if item["type"] == CellType.BALL
            ),
            None
        )
        if ball is not None:
            return RewardEvent.REACH_BALL
        return None
    
    
    def _R4_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ PICK_UP_ITEM BALL successfully: +2.00 """
        if action != Action.PICK_UP_ITEM:
            return None
        state_pos = (state["i"], state["j"])
        if state["has_ball"] is not False:
            return None
        if len(state["snapshot"]["inventory"]) >= AGENT_MAX_LOAD:
            print("_R4_event reward denied for violation of AGENT_MAX_LOAD")
            return None
        state_cell_inventory = state["snapshot"]["objects"].get(state_pos, [])
        ball = next(
            (
                item for item in state_cell_inventory
                if item["type"] == CellType.BALL
            ),
            None
        )
        if ball is None:
            return None
        ball_removed = not any(  item["type"] == CellType.BALL 
                                 for item in next_state["snapshot"]["objects"].get(state_pos, []) ) 
        if (
            next_state["has_ball"] is True
            and next_state["door_state"] == DoorState.UNBLOCKED
            and ball_removed 
        ):
            return RewardEvent.PICKUP_BALL
        return None

    def _R4r_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ REPEAT PICK_UP_ITEM on BALL after PICKUP_BALL once: -0.01 """
        if action != Action.PICK_UP_ITEM:
            return None
        # successful pickup transition
        if not (state["has_ball"] is False and next_state["has_ball"] is True):
            return None
        # first meaningful pickup
        if RewardEvent.PICKUP_BALL not in state["snapshot"]["visited_events"]:
            return None
        # only after ball objective already completed
        if state["door_state"] in [DoorState.UNBLOCKED, DoorState.OPEN]:
            return RewardEvent.REPEAT_PICKUP_BALL
        return None
    
    
    def _R5_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """Valid DROP_ITEM for the ball (away from the door): +3.00 """
        if action != Action.DROP_ITEM:
            return None
        if state["has_ball"] is not True:
            return None
        state_pos = (state["i"], state["j"])
        state_cell_inventory = state["snapshot"]["objects"].get(state_pos, [])
        if len(state_cell_inventory) >= ENV_MAX_LOAD:
            print("_R5_event reward denied for violation of ENV_MAX_LOAD")
            return None
        door_pos = state["snapshot"]["doors"][self.door_color]["pos"]
        if state_pos == door_pos:
            return None
        manhattan_distance = (
            abs(state_pos[0] - door_pos[0]) +
            abs(state_pos[1] - door_pos[1])
        )
        if manhattan_distance < 2:
            return None
        next_state_pos = (next_state["i"], next_state["j"])
        next_state_cell_inventory = next_state["snapshot"]["objects"].get(next_state_pos, [])
        ball_exists = any(
            item["type"] == CellType.BALL
            for item in next_state_cell_inventory
        )
        if next_state["has_ball"] is False and ball_exists:
            return RewardEvent.DROP_BALL_VALID
        return None


    def _R5d_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ DROP_ITEM BALL on the cell next to the left of the door: -1.00 """
        if action != Action.DROP_ITEM:
            return None
        if state["has_ball"] is not True:
            return None
        state_pos = (state["i"], state["j"])
        door_pos = state["snapshot"]["doors"][self.door_color]["pos"]
        door_row, door_col = door_pos
        if state_pos == door_pos:
            return None
        manhattan_distance = (
            abs(state_pos[0] - door_row) +
            abs(state_pos[1] - door_col)
        )
        if manhattan_distance > 2:
            return None
        # ONLY left side of the door
        if state_pos != (door_row, door_col - 1):
            return None
        next_state_pos = (next_state["i"], next_state["j"])
        next_state_cell_inventory = next_state["snapshot"]["objects"].get(next_state_pos, [])
        ball_exists_in_cell = any(
            item["type"] == CellType.BALL
            for item in next_state_cell_inventory
        )
            
        if (next_state["has_ball"] is False 
            and ball_exists_in_cell
            and next_state["door_state"] == DoorState.LOCKED):
            return RewardEvent.DROP_BALL_INVALID
        return None

    ## --------------- Key Related Rewards ---------------------------------------
    
    def _R6_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """Reaching a cell for the first time with the correct key: +1.00"""
        allowed_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        if state["has_key"] is not None:
            return None
        if action not in allowed_actions:
            return None
        next_state_pos = (next_state["i"], next_state["j"])
        if next_state_pos in self.visited_cells:
            return None
        next_state_cell_inventory = next_state["snapshot"]["objects"].get(next_state_pos, [])
        key = next(
            (item for item in next_state_cell_inventory
             if item["type"] == CellType.KEY),
            None
        )
        if key is None:
            return None
        if key["color"] == self.door_color:
            return RewardEvent.REACH_KEY_CORRECT
        return None

    
    def _R6d_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """R6d Arriving at a cell for the first time with the wrong key:  +0.00 """
        allowed_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        if state["has_key"] is not None:
            return None
        if action not in allowed_actions:
            return None
        next_state_pos = (next_state["i"], next_state["j"])
        if next_state_pos in self.visited_cells:
            return None
        next_state_cell_inventory = next_state["snapshot"]["objects"].get(next_state_pos, [])
        key = next(
            (item for item in next_state_cell_inventory
             if item["type"] == CellType.KEY),
            None
        )
        if key is None:
            return None
        if key["color"] != self.door_color:
            return RewardEvent.REACH_KEY_WRONG
        return None

    
    def _R7_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ PICK_UP_ITEM Successful on right key:: +2.00"""
        if action != Action.PICK_UP_ITEM:
            return None
        if state["has_key"] is not None:
            return None
        if len( state["snapshot"]["inventory"]) >= AGENT_MAX_LOAD:
            print("_R7_event reward denied for violation of AGENT_MAX_LOAD")
            return None
        state_pos = (state["i"], state["j"])    
        state_cell_inventory =  state["snapshot"]["objects"].get(state_pos, [])
        key = next(
            (item for item in state_cell_inventory
             if item["type"] == CellType.KEY),
            None
        )
        if key is None:
            return None
        if key["color"] != self.door_color:
            return None
        key_removed = not any( item["type"] == CellType.KEY 
                               and item["color"] == key["color"] 
                               for item in next_state["snapshot"]["objects"].get(state_pos, []) ) # Comment if issue
        if not key_removed:
            print(f"_R7_event reward denied for violation of item removal, key still exist in cell {state_pos} in next_state")
            return None
        if (next_state["has_key"] == self.door_color):
            return RewardEvent.PICKUP_KEY_CORRECT
        return None
    
    
    def _R7d_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ PICK_UP_ITEM Successful on incorrect key: -0.50"""
        if action != Action.PICK_UP_ITEM:
            return None
        if len(state["snapshot"]["inventory"]) >= AGENT_MAX_LOAD:
            print("_R7d_event reward denied for violation of AGENT_MAX_LOAD")
            return None
        state_pos = (state["i"], state["j"])
        state_cell_inventory = state["snapshot"]["objects"].get(state_pos, [])
        key = next(
            (item for item in state_cell_inventory
             if item["type"] == CellType.KEY),
            None
        )
        if key is None:
            return None
        if key["color"] == self.door_color:
            return None
        key_removed = not any(
            item["type"] == CellType.KEY and item["color"] == key["color"]
            for item in next_state["snapshot"]["objects"].get(state_pos, [])
        )
        if next_state["has_key"] == key["color"] and key_removed:
            return RewardEvent.PICKUP_KEY_WRONG
        return None
    
    
    ## --------------- Door Related Rewards ---------------------------------------
    
    def _R8_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ Reach the cell at the left of the unlocked door: +2.00"""
        allowed_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        if action not in allowed_actions:
            return None
        if state["door_state"] != DoorState.UNBLOCKED:
            return None
        if RewardEvent.REACH_DOOR_UNBLOCKED in state["snapshot"]["visited_events"]:
            return None
        door_pos = state["snapshot"]["doors"][self.door_color]["pos"]
        target_pos = (door_pos[0], door_pos[1] - 1)
        next_state_pos = (next_state["i"], next_state["j"])
        if next_state_pos == target_pos:
            return RewardEvent.REACH_DOOR_UNBLOCKED
        return None
    
    
    def _R9_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ Open door successfully: +3.00"""
        if action != Action.USE_ITEM:
            return None
        if state["door_state"] != DoorState.UNBLOCKED:
            return None
        if state["has_key"] != self.door_color:
            return None
        door_pos = state["snapshot"]["doors"][self.door_color]["pos"]
        required_pos = (door_pos[0], door_pos[1] - 1)
        state_pos = (state["i"], state["j"])
        if state_pos != required_pos:
            return None
        if next_state["door_state"] == DoorState.OPEN:
            return RewardEvent.OPEN_DOOR
        return None
    
    
    def _R10_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ Move through the door into the room on the right: +2.00 """
        if action != Action.RIGHT:
            return None
        if state["door_state"] != DoorState.OPEN:
            return None
        if state["left_room"] is not False:
            return None
        door_pos_j = state["snapshot"]["doors"][self.door_color]["pos"][1]
        if next_state["j"] > door_pos_j and next_state["left_room"] is True:
            return RewardEvent.CROSS_DOOR_TO_RIGHT
        return None
    
    
    def _R14_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ Penalty for returning to the left room: -2.00"""
        if action != Action.LEFT:
            return None
        if state["left_room"] is not True:
            return None
        door_pos_j = state["snapshot"]["doors"][self.door_color]["pos"][1]
        if state["j"] != door_pos_j + 1:
            return None
        if next_state["j"] < door_pos_j and next_state["left_room"] is False:
            return RewardEvent.PENALTY_BACKTRACK_TO_LEFT
        return None
    
    
    ## --------------- Goal / Global Rewards ---------------------------------------
    
    def _R11_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ Successful EXIT at cell END: +10.00"""
        if action != Action.EXIT:
            return None
        state_pos = (state["i"], state["j"])
        if state_pos == self.end_state_pos:
            return RewardEvent.EXIT_GOAL
        return None
    
    
    def _R12_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ Invalid action: -0.50"""
        allowed_actions = self.get_possible_actions(state)
        if action not in allowed_actions:
            return RewardEvent.INVALID_ACTION
        return None
    
    
    def _R15_event(self, state: dict, action: Action, next_state: dict) -> RewardEvent:
        """ Timeout  MAX_STEPS: -5.00 """
        if state["steps"] >= MAX_STEPS:
            return RewardEvent.TIMEOUT
        return None
        


        
    def _detect_events(self, state: dict, action: Action, next_state: dict) -> list[RewardEvent]:
        """Returns the events detected when executing that action in the current state,
        with respect to the new state generated."""
        event_detectors = [
            self._R1_event,
            self._R2_event,
            self._R3_event,
            self._R4_event,
            self._R4r_event,
            self._R5_event,
            self._R5d_event,
            self._R6_event,
            self._R6d_event,
            self._R7_event,
            self._R7d_event,
            self._R8_event,
            self._R9_event,
            self._R10_event,
            self._R11_event,
            self._R12_event,
            self._R13_event,
            self._R14_event,
            self._R15_event,
        ]
        detected_events = []
        for detector in event_detectors:
            event = detector( state, action, next_state )
            if event is not None:
                detected_events.append(event)
    
        return detected_events

    def _compute_reward(self, state: dict, action: Action, next_state: dict, event: RewardEvent) -> float:
        """ Single source of truth for rewards.
        Parameters
        ----------
        state     : state before the action
        action    : executed action
        next_state: resulting state
        event     : event that occurred (see REWARD_TABLE)
        
        Returns float: obtained reward (0.0 if the event has already been claimed and is 'once')
        """
        if event is None:
            return 0.0
        if not isinstance(event, RewardEvent):
            return 0.0
        if event.once:
            if event in state["snapshot"]["visited_events"]:
                return 0.0          # ya se cobró, no se repite
            self.visited_events.add(event)
        return event.reward

        
    def do_action(self, action:Action) -> tuple[float, dict,  bool, dict]:
        """Executes the action and modifies the actual state of the environment.
        Returns (reward, new_state, done, info) in all cases—even for invalid actions,
        in which case it returns (reward=-0.5, current_state_unchanged, False, {'event':[RewardEvent.INVALID_ACTION]}).
        It does NOT raise exceptions for invalid actions; instead, it penalizes them with R12 INVALID_ACTION."""
        state = self.get_current_state(include_snapshot=True)
        
        #  --------------- Validar action ---------------
        valid_actions = self.get_possible_actions(state)
        if action not in valid_actions:
            reward = self._compute_reward(state, action, state, RewardEvent.INVALID_ACTION)
            return (reward, state, False, {"events": [RewardEvent.INVALID_ACTION] })
            
        # --------- Aplicar transición -> nuevo estado ---------------
        transition_data = self._apply_transition(state, action)

        steps = self.steps + 1
        next_state = self._compute_state(
            state_pos       = transition_data["state_pos"],
            inventory       = transition_data["inventory"],
            doors           = transition_data["doors"],
            steps           = steps,
            door_color      = state["door_color"],
            objects         = transition_data["objects"],
            include_snapshot= True # Necesaary for _detect_events to get the pre-transition snapshot 
        )
        
        # --------- Detectar eventos ---------------
        events = self._detect_events(state, action, next_state)
        
        # --------- y calcular reward ---------------
        total_reward = sum(self._compute_reward(state, action, next_state, event)
                          for event in events)

        # ------------ Update internal state ------------
        # --------------------------------------------------------
        self.state            = transition_data["state_pos"]
        self.agent_inventory  = transition_data["inventory"]
        self.objects          = transition_data["objects"]
        self.doors            = transition_data["doors"]
        self.steps            = steps
        
        # ------------ Verificar terminal ---------------
        done = ( 
            RewardEvent.EXIT_GOAL in events 
            or RewardEvent.TIMEOUT in events 
            or self.is_terminal(next_state) 
        )

        # ------------ Update visited cells -----------------------------
        next_state_pos = ( next_state["i"], next_state["j"] )
        self.visited_cells.add(next_state_pos)

        del transition_data
        
        return total_reward, next_state, done, {"events": events}


    def step(self, action:Action) -> tuple[float, dict,  bool, dict]:
        """Standard RL interface to wrap do_action to work with Gym conventions."""
        reward, state, done, info = self.do_action(action)
        return state, reward, done, info
        

    def _apply_transition(self, state:dict, action:Action) -> dict:
        """Executes the action and computes the transition to the new state,
        WITHOUT mutating the actual environment.
        Returns transition results."""
        i, j      = state["i"], state["j"]
        state_pos = (i, j)
        agent_inventory = deepcopy(state["snapshot"]["inventory"] )#deepcopy(self.agent_inventory)
        objects   = deepcopy(state["snapshot"]["objects"]) #deepcopy(self.objects)
        doors     = deepcopy(state["snapshot"]["doors"]) #deepcopy(self.doors)
        door_color = deepcopy(state["door_color"])
            
        if action == Action.UP: #and i > 0 and self.grid[i-1][j] != None: # Uncoment to restrit agent
            i -= 1
        elif action == Action.DOWN: # and i < self.nrows-1 and self.grid[i+1][j] != None: # Uncoment to restrit agent
            i += 1
        elif action == Action.LEFT: # and j > 0 and self.grid[i][j-1] != None: # Uncoment to restrit agent
            j -= 1
        elif action == Action.RIGHT: #and j < self.ncols-1 and self.grid[i][j+1] != None: # Uncoment to restrit agent
            j += 1
            
        elif action == Action.PICK_UP_ITEM:
            cell_objects = objects.get( state_pos, [] )
            if (len(cell_objects) > 0
               and len(agent_inventory) < AGENT_MAX_LOAD):
                picked_item = cell_objects.pop(0)
                agent_inventory.append( picked_item )
    
                ## Ball unblocks door
                if (picked_item["type"] == CellType.BALL 
                    and doors[door_color]["door_state"] == DoorState.LOCKED 
                    and doors[door_color]["pos"][0] == i
                    and doors[door_color]["pos"][1] == ( j + 1 ) # right next to the door
                   ):
                    doors[door_color][ "door_state" ] = DoorState.UNBLOCKED
                objects[state_pos] = cell_objects
                
        elif action == Action.USE_ITEM: 
             if ( doors[door_color]["pos"][0] == i
                  and doors[door_color]["pos"][1] == ( j + 1 ) # right next to the door,
                  and state["has_key"] == door_color
                  and doors[door_color]["door_state"] == DoorState.UNBLOCKED
                ):
                 doors[door_color][ "door_state" ] = DoorState.OPEN
            
        elif action == Action.DROP_ITEM:
             if len(agent_inventory) > 0:
                dropped_item = agent_inventory.pop(0)
                cell_objects = objects.get( state_pos, [] )
                cell_objects.append(dropped_item)
                objects[state_pos] = cell_objects

                # block the door again if drop the ball next to the door
                if (dropped_item["type"] == CellType.BALL 
                    and doors[door_color]["pos"][0] == i
                    and doors[door_color]["pos"][1] == ( j + 1 ) # right next to the door
                   ):
                    doors[door_color][ "door_state" ] = DoorState.LOCKED
                 
        elif action == Action.EXIT:
            if state_pos == self.end_state_pos:
                #print("EXIT")
                pass

        return {
            "state_pos": (i, j),
            "inventory": agent_inventory,
            "objects": objects,
            "doors": doors,
        }


    def reset(self, new_board = None):
        """Resets the environment to its initial state.
        Restores: object positions to the original board configuration, self.door['state']='locked',
        self.agent_inventory=[], self.visited_cells=set(), self.visited_events=set(), self.steps=0.
        Returns the complete initial state."""
        del self.objects
        del self.doors
        del self.agent_inventory
        del self.visited_cells
        del self.visited_events
        
        self.objects = {}
        self.doors = {}
        self.agent_inventory = []
        self.visited_cells = set()
        self.visited_events = set()
        self.steps = 0
        ## Re-encode original board
        if new_board is None:
            new_board = self.board_copy
            new_dimensions =  (len(self.board_copy), len(self.board_copy[0]) )
        else:
            new_dimensions = (len(new_board), len(new_board[0]) )
            
        self.nrows, self.ncols = new_dimensions
        super().__init__(new_board, new_dimensions)
        
        #self.encode(new_board)
        self.state = self.initial_state
        return self.get_current_state()
        
    
    def is_terminal(self, state:tuple[int,int]=None) -> bool:
        """Retorna True si: (a) el último do_action ejecutó EXIT en celda END,
        o (b) self.steps >= MAX_STEPS."""
        if state is None:
            state = self.get_current_state(include_snapshot=True)

        # ---- Timeout ----------------
        if state["steps"] >= MAX_STEPS:
            return True
        # ---- Agent at END and EXIT available ----
        state_pos = ( state["i"], state["j"] )
    
        if state_pos == self.end_state_pos and Action.EXIT in self.get_possible_actions(state):
            return False # Change to True in case we don't want to wait for the agent to dicide whe to EXIT
    
        return False
        
    def render(self, mode: str = 'human') -> None:
        """Visualizes the current state of the environment in the console."""
        # Build empty canvas
        canvas = [
            ["." for _ in range(self.ncols)]
            for _ in range(self.nrows)
        ]
        # Draw walls
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.grid[i][j] == CellType.WALL.code:
                    canvas[i][j] = "#"
        # Draw goal
        gi, gj = self.end_state_pos
        canvas[gi][gj] = "G"
        # Draw door
        door = self.doors[self.door_color]
        di, dj = door["pos"]
    
        if door["door_state"] == DoorState.LOCKED:
            canvas[di][dj] = "D"
        elif door["door_state"] == DoorState.UNBLOCKED:
            canvas[di][dj] = "U"
        elif door["door_state"] == DoorState.OPEN:
            canvas[di][dj] = "O"
        # Draw objects (keys, balls)
        for (i, j), items in self.objects.items():
            if not items:
                continue
            # prioritize visibility
            for item in items:
                if item["type"] == CellType.KEY:
                    canvas[i][j] = "K"
                elif item["type"] == CellType.BALL:
                    canvas[i][j] = "B"
        # Draw agent (always last so it overrides cell)
        ai, aj = self.state
        canvas[ai][aj] = "A"
        # Print metadata (very useful for RL debugging)
        print("\n" + "=" * 40)
        print(f"Step: {self.steps}")
        print(f"Door state: {self.doors[self.door_color]['door_state']}")
        print(f"Inventory: {self.agent_inventory}")
        print(f"Visited cells: {len(self.visited_cells)}")
        print("=" * 40)
        # Print grid
        for row in canvas:
            print(" ".join(row))
        print("\n")

    
    def simulate_action(self, state: tuple, action: str) -> tuple:
        """Simulates a step WITHOUT modifying any environment attributes (pure method).
        Returns (reward, new_state, done, info).
        Applies validate_restrictions internally.
        Useful for planning and debugging without altering the actual state."""
        raise NotImplementedError("Subclasses must implement simulate_action method!")
    

    def run_golden_episode(self, actions: list[Action], verbose: bool = True):
        """
        Runs a deterministic episode to validate:
        - transitions
        - rewards
        - events
        - step consistency
        """
        state = self.reset()
        total_reward = 0.0
        if verbose:
            print("\n" + "=" * 60)
            print("🔬 GOLDEN EPISODE TEST START")
            print("=" * 60)
        for t, action in enumerate(actions):
            prev_state = state
            reward, state, done, info = self.do_action(action)
            total_reward += reward
            if verbose:
                print(f"\nStep {t}")
                print(f"Action: {action}")
                print(f"Reward: {reward:.3f}")
                print(f"Done: {done}")
                print(f"Events: {info['events']}")
    
                print(f"Pos: ({state['i']}, {state['j']})")
                print(f"Door: {state['door_state']}")
                print(f"Key: {state['has_key']}")
                print(f"Ball: {state['has_ball']}")
                print(f"Steps: {state['steps']}")
                print(f"Visited cells: {len(self.visited_cells)}")
            # ---- HARD CHECKS (debug safety) ----
            assert state is not None, "State became None"
            assert isinstance(state, dict), "State must be dict"
            assert "i" in state and "j" in state, "Missing position keys"
            if done:
                if verbose:
                    print("\n🏁 Episode finished early (done=True)")
                break
        if verbose:
            print("\n" + "=" * 60)
            print("TOTAL REWARD:", total_reward)
            print("EPISODE END")
            print("=" * 60)
    
        return total_reward

        