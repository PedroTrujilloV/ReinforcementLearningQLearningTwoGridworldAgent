from enum import Enum

# ── REWARD_TABLE ─────────────────────────────────────────────────────────────
# Fuente única de verdad para todas las recompensas del ambiente.
# Se pasa como parámetro al constructor del MDP y es consultada por _compute_reward.
#
# Estructura de cada entrada:
#   "event_key": (reward: float, once: bool)
#
#   - reward : valor numérico de la recompensa (positivo) o penalización (negativo)
#   - once   : True  -> solo se cobra la primera vez en el episodio (tracked en visited_events)
#              False -> se cobra en cada ocurrencia
#
# Convención de nombres:
#   reach_*   -> el agente llega a una celda por primera vez
#   pickup_*  -> el agente ejecuta PICK_UP_ITEM exitosamente
#   drop_*    -> el agente ejecuta DROP_ITEM exitosamente
#   use_*     -> el agente ejecuta USE_ITEM exitosamente
#   cross_*   -> el agente cruza un umbral (puerta, cuarto)
#   invalid_* -> acción no aplicable en el estado actual
#   penalty_* -> penalización continua por condición activa
# ─────────────────────────────────────────────────────────────────────────────

REWARD_TABLE = {

    # ── R1: Penalización por paso ─────────────────────────────────────────────
    # Incentiva eficiencia. Sin esto el agente puede vagar indefinidamente.
    # Debemos considerar el tamaño del tablero para poder sacar un buen estimado
    # Para este reward porque por ejemplo: 
    # tablero (3x7): new_cell reward - step_penalty =  (3*7 * 0.5) +  (-0.01 * 500)
    # = 5
    # tablero (4x9): new_cell reward - step_penalty =  (4*9 * 0.5) + (-0.01 * 500)
    # = 13
    # Lo que corresponde a los resultados de los experimentos, donde el agente
    # aprende mas mejor y mas rapido en ambietes con tableros pequeños 
    # y tiende a vagar mas en ambientes con tableros grandes y a conformaarse con 
    # recompensaas de ~ +6.0
    "step_penalty":             (-0.02, False), # Antes (-0.01, False), 

    # ── R2: Exploración celda nueva ─────────────────────────────────────────
    # Se cobra una vez por celda única visitada (tracked en visited_cells).
    # Valor menor que reach_ball/reach_key para crear gradiente de incentivos.
    "new_cell":                 ( 0.10, True), # Antes ( 0.50, True),

    # ── R3: Llegar a la bola ──────────────────────────────────────────────────
    # Primera vez que el agente pisa la celda que contiene la bola.
    "reach_ball":               ( 1.00, True), # Antes ( 1.00, True),

    # ── R4: Recoger la bola ───────────────────────────────────────────────────
    # PICK_UP_ITEM exitoso sobre objeto type="ball".
    # Implica que door["state"] cambia de 'locked' -> 'unblocked'.
    "pickup_ball":              ( 2.00, True),

    # ── R4: Recoger la bola de nuevo ───────────────────────────────────────────
    # PICK_UP_ITEM sobre objeto type="ball".
    # Si door["state"] esta en 'open' o en 'unblocked'.
    # Proposito: Desmotivar al agente de repetir esta accion multiples veces
    # Ya que solo es necesaria una ves en el actual escenario
    "repeat_pickup_ball":       ( 0, False), # No utilizado 

    # ── R5: Tirar la bola en posición válida ──────────────────────────────────
    # DROP_ITEM de la bola en celda con distancia Manhattan ≥ 1 desde door["pos"],
    # y que la celda destino no sea tipo 2 (DOOR_LOCKED) ni tipo 3 (DOOR_OPEN).
    "drop_ball_valid":          ( 3.00, True),

    # ── R5d: Penalización por tirar la bola junto a la puerta  ──
    #  DROP_ITEM de la bola en celda a la izquierda desde door["pos"],
    # y si door["state"] ==  (DOOR_OPEN) (auto bloquearse). 
    # Esto hará que el estado de la puerta cambie a DOOR_LOCKED
    "drop_ball_invalid":        (-3.00, False),# Antes (-1.00, False),

    # ── R6: Llegar a la llave correcta ───────────────────────────────────────
    # Primera vez que el agente pisa la celda de una llave cuyo color
    # coincide con door["color"]. key["color"] == door["color"].
    "reach_key_correct":        ( 2.00, True),

    # ── R6d: Llegar a llave distractor ───────────────────────────────────────
    # Primera vez que el agente pisa la celda de una llave cuyo color
    # NO coincide con door["color"]. No recompensa ni penaliza la visita,
    # pero sí se cobra R2 (new_cell) por ser celda nueva.
    "reach_key_wrong":          ( 0.00, True),

    # ── R7: Recoger la llave correcta ────────────────────────────────────────
    # PICK_UP_ITEM exitoso sobre llave con key["color"] == door["color"].
    "pickup_key_correct":       ( 2.00, True),

    # ── R7d: Recoger llave distractor ────────────────────────────────────────
    # PICK_UP_ITEM sobre llave con key["color"] != door["color"].
    # Penaliza para que el agente aprenda a ignorar distractores.
    # once=False porque el agente podría soltar y volver a recoger el distractor.
    "pickup_key_wrong":         (-0.50, False),

    # ── R8: Llegar a la celda adyacente a la puerta (desbloqueada) ────────────
    # Primera vez que el agente llega a (door["pos"][0], door["pos"][1] - 1)
    # con door["state"] == 'unblocked'. Si llega antes de retirar la bola,
    # este evento NO se cobra (condición evaluada en _compute_reward).
    "reach_door_unblocked":     ( 3.00, True), #  Antes ( 2.00, True),

    # ── R9: Abrir la puerta ───────────────────────────────────────────────────
    # USE_ITEM exitoso: has_key == door["color"] y door["state"] == 'unblocked'.
    # door["state"] cambia a 'open'.
    "open_door":                ( 10.00, True), # Antes  ( 3.00, True),

    # ── R10: Cruzar la puerta al cuarto derecho ───────────────────────────────
    # Primer paso del agente en el cuarto derecho (j > door["pos"][1]).
    # left_room cambia a True.
    "cross_door_to_right":      ( 8.00, True), # Antes  ( 2.00, True),

    # ── R11: Completar el objetivo (EXIT) ─────────────────────────────────────
    # Agente ejecuta EXIT en celda tipo 4 (END). Episodio termina con done=True.
    "exit_goal":                (100.00, True), # Antes (10.00, True), 

    # ── R12: Acción inválida ──────────────────────────────────────────────────
    # validate_restrictions retorna False. Cubre: colisión con pared/borde,
    # USE_ITEM con llave incorrecta, PICK_UP_ITEM con inventario lleno,
    # DROP_ITEM en celda de puerta, EXIT fuera de celda objetivo, etc.
    "invalid_action":           (-0.50, False),

    # ── R13: Penalización por permanecer en cuarto izquierdo post-apertura ────
    # Cada paso en el cuarto izquierdo (j < door["pos"][1]) después de que
    # door["state"] == 'open'. Incentiva cruzar la puerta sin demora.
    "penalty_left_room_after_open": (-2.00, False), # Antes  (-1.00, False),

    # ── R14: Penalización por regresar al cuarto izquierdo ────────────────────
    # El agente tenía left_room=True y ejecuta LEFT desde la celda adyacente
    # derecha de la puerta, regresando al cuarto izquierdo.
    # left_room vuelve a False.
    "penalty_backtrack_to_left":    (-2.00, False),

    # ── R15: Timeout ──────────────────────────────────────────────────────────
    # steps >= MAX_STEPS sin haber ejecutado EXIT. is_terminal() retorna True.
    "timeout":                  (-50.00, True), # Antes (-5.00, True),

}



class RewardEvent(Enum):
    """
    Enumeration of all reward and penalty events.

    Each enum value stores the string event key.
    Reward metadata is retrieved dynamically from REWARD_TABLE.
    """

    STEP_PENALTY = "step_penalty"

    NEW_CELL = "new_cell"

    REACH_BALL = "reach_ball"

    PICKUP_BALL = "pickup_ball"

    REPEAT_PICKUP_BALL = "repeat_pickup_ball"

    DROP_BALL_VALID = "drop_ball_valid"
    
    DROP_BALL_INVALID = "drop_ball_invalid"
    
    REACH_KEY_CORRECT = "reach_key_correct"

    REACH_KEY_WRONG = "reach_key_wrong"

    PICKUP_KEY_CORRECT = "pickup_key_correct"

    PICKUP_KEY_WRONG = "pickup_key_wrong"

    REACH_DOOR_UNBLOCKED = "reach_door_unblocked"

    OPEN_DOOR = "open_door"

    CROSS_DOOR_TO_RIGHT = "cross_door_to_right"

    EXIT_GOAL = "exit_goal"

    INVALID_ACTION = "invalid_action"

    PENALTY_LEFT_ROOM_AFTER_OPEN = "penalty_left_room_after_open"

    PENALTY_BACKTRACK_TO_LEFT = "penalty_backtrack_to_left"

    TIMEOUT = "timeout"

    @property
    def reward(self) -> float:
        return REWARD_TABLE[self.value][0]

    @property
    def once(self) -> bool:
        return REWARD_TABLE[self.value][1]

    @property
    def description(self) -> str:

        descriptions = {

            RewardEvent.STEP_PENALTY:
                "Penalty applied every step to encourage efficiency.",

            RewardEvent.NEW_CELL:
                "Reward for visiting a new cell for the first time.",

            RewardEvent.REACH_BALL:
                "Reward for reaching the cell containing the ball.",

            RewardEvent.PICKUP_BALL:
                "Reward for successfully picking up the ball.",

            RewardEvent.REPEAT_PICKUP_BALL:
                "Penalty for picking up the ball after door becoming unblocked or open",

            RewardEvent.DROP_BALL_VALID:
                "Reward for dropping the ball in a valid position.",
            
            RewardEvent.DROP_BALL_INVALID:
                "Penalty for dropping the ball in the cell at the left of the door (self blocking penalty).",

            RewardEvent.REACH_KEY_CORRECT:
                "Reward for reaching the correct key.",

            RewardEvent.REACH_KEY_WRONG:
                "Neutral event for reaching a distractor key.",

            RewardEvent.PICKUP_KEY_CORRECT:
                "Reward for picking up the correct key.",

            RewardEvent.PICKUP_KEY_WRONG:
                "Penalty for picking up a distractor key.",

            RewardEvent.REACH_DOOR_UNBLOCKED:
                "Reward for reaching the door after it becomes unblocked.",

            RewardEvent.OPEN_DOOR:
                "Reward for successfully opening the door.",

            RewardEvent.CROSS_DOOR_TO_RIGHT:
                "Reward for entering the right room for the first time.",

            RewardEvent.EXIT_GOAL:
                "Reward for successfully exiting the environment.",

            RewardEvent.INVALID_ACTION:
                "Penalty for attempting an invalid action.",

            RewardEvent.PENALTY_LEFT_ROOM_AFTER_OPEN:
                "Penalty for staying in the left room after the door is open.",

            RewardEvent.PENALTY_BACKTRACK_TO_LEFT:
                "Penalty for returning to the left room after crossing.",

            RewardEvent.TIMEOUT:
                "Penalty applied when the episode reaches the maximum number of steps.",
        }

        return descriptions[self]

