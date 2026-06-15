from IPython.display import Video
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import copy
from copy import deepcopy
from Actions import Action
from EnvironmentProperties import CellType, Color, DoorState
from Reward import RewardEvent
from EpisodeRecorder import EpisodeRecorder
from EpisodeRecorder import SpriteEpisodeRecorder
from QLearningPlotter import QLearningPlotter
from BoardGenerator import BoardGenerator
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings("ignore", message=".*probesize.*")

try:
    import imageio
    import imageio_ffmpeg
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "imageio", "imageio-ffmpeg"], check=True)
    #!pip install imageio imageio-ffmpeg #in case subprocess doesn't work.
    import imageio
    
class QLearningAgent:
    def __init__(self, 
                 environment, gamma:float=0.9, alpha:float=0.5,
                 epsilon=0.9, epsilon_decay:float = 0.995, epsilon_min: float = 0.01,
                 episodes:int = 1000, samples:int = 0, explore:int = 0, threshold:float = 0.00001, 
                 seed = 7):
    
        self.env = environment
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = self.epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.episodes = episodes
        self.samples = samples
        self.explore = explore
        self.threshold = threshold
        
        self.Q_table = {} # {(state, action):  Q-value}
        self.policy: dict[tuple, Action] = {} # Greedy policy derived from Q_table: {state: best_action}
        self.rewards_history: list[float] = [] # Total reward per episode (for training plots)
        self.steps_history:list[int] = [] #Number of steps per episode
        self.epsilon_history:list[float] = [] # for debugging exploration collapse.
        self.td_error_history:list[float] = [] # for diagnose convergence by storing the Temporal Difference (TD) Error: 𝛾max𝑎′𝑄(𝑠′,𝑎′)−𝑄(𝑠,𝑎)
        random.seed(seed)
        np.random.seed(seed)

    
    def get_value(self, state:dict, action:Action) -> float:
        state_key = self.encode_state(state)
        return self.Q_table.get((state_key, action), 0)

    def encode_state(self, state: dict) -> tuple:
        "Codifica el estado para que cumpla con la propiedad de Markov"
        return (
            state["i"],
            state["j"],
            state["has_key"],
            state["has_ball"],
            state["door_state"],
            state["door_color"],
            state["left_room"],
            # The reason why steps is not here is because it will explode the state space 
            # this means that the next time the agent finds (1,0,...,steps=5) and (1,0,...,steps=6) become DIFFERENT states,
            # so the agent never reuses knowledge. Never learns
        )
        
    # ------- choose_action ----------------------------------------------------------------------------
    def choose_action(self, state: dict) -> Action:
        """Política ε-greedy: con probabilidad ε elige acción aleatoria válida,
        con probabilidad 1-ε elige la acción con mayor Q(s,a).
        Solo considera acciones válidas según get_possible_actions(state)."""
        # epsilon-greedy con selección del mejor brazo
        if random.random() < self.epsilon:
            # Exploración: probabilidad epsilon
            possible_actions = self.env.get_possible_actions(state)
            return random.choice(possible_actions)
        else:
            # Explotación: mayor Q(b) con probabilidad 1-epsilon
            # np.argmax retorna el índice del máximo; si hay empate, retorna el primero
            # Para retornar cualquiera en caso de empate, filtramos los índices con valor máximo
            return self.best_action(state)
            
    # ------- update_Q ----------------------------------------------------------------------------
    
    def update_Q(self, state:dict, action:Action, reward:float, next_state:dict, done: bool) -> float:
        """Actualización de Bellman para Q-Learning:
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a'(Q(s',a')) - Q(s,a)]
        Si done=True, el término de bootstrap se omite (Q terminal = 0)."""
        # 𝑄(𝑠,𝑎)=(1−𝛼)𝑄(𝑠,𝑎)+𝛼[𝑟+𝛾max𝑎′𝑄(𝑠′,𝑎′)] = 𝑄(𝑠,𝑎)=𝑄(𝑠,𝑎)+𝛼[𝑟+𝛾max𝑎′𝑄(𝑠′,𝑎′)−𝑄(𝑠,𝑎)]
        next_action =  self.best_action(next_state)
        if done:   
            max_Q_sa2 = 0  # For terminal states: a′max Q(s′,a′)= 0
        elif next_action is None:
            max_Q_sa2 = 0  
        else:
            max_Q_sa2 = self.get_value(next_state, next_action) 
        # -------- Target compute --------
        target = reward + self.gamma * max_Q_sa2
        Q_sa1 =  self.get_value(state, action)
        # -------- TD Error --------
        td_error = target - Q_sa1
        # -------- Q update --------
        new_Q = Q_sa1 + self.alpha * td_error
        # -------- Store --------
        state_key = self.encode_state(state)
        self.Q_table[(state_key, action)] = new_Q
        # -------- Debug metrics --------
        
        return abs(td_error)

    # ------- best_action ----------------------------------------------------------------------------
    
    def best_action(self, state:dict) -> Action:
        """Recibe un estado por parámetro y retorna la mejor acción a ejecutar para dicho estado. 
        La mejor acción para un estado corresponde a la acción con mayor q-valor para el estado.
        El resultado de la función debe ser el nombre de la mejor acción para el estado."""
        possible_actions = self.env.get_possible_actions(state)
        if len(possible_actions) == 0:
            return None
        q_values = [(action, self.get_value(state, action)) for action in possible_actions]
        max_value = max(v for _, v in q_values)
        best_actions = [a for a, v in q_values if v == max_value]
        return random.choice(best_actions)

    
    def step(self, action: Action) -> tuple[float, dict,  bool, dict]:
        """se encarga de ejecutar un paso del agente (una acción). 
        Dada la acción a ejecutar, esta función la ejecuta dentro del ambiente"""
        return self.env.step(action)

    # ------- train ----------------------------------------------------------------------------

    def train(self, recorder = None, n_records = 10 , randomized_boards = None, verbose = True) -> dict:
        """Ejecuta el loop de entrenamiento por `episodes` episodios.
        Parametros:
            recorder : EpisodeRecorder = Video Recorder 
            n_records: Cantidad deseada de episodios grabados en video. 
                       por defecto se deja = 10 para optimizar por defecto
                       Si este parametro es None o = episodes, se graba todo el entrenamiento,
                       Lo que consume mas tiempo y recursos, 
                       n_records es min( n_records, episodios) y max(1, n_records)
                       funcional solo si recorder.
                       Si es = 1, solo graba el episodio final.
            randomized_boards: tableros de ejemplo sobre los cuales se generaran tablero aleatorios de entrenamiento
                               si este prametro es None, se entrena sobre el mismo tablero.
                               
        Retorna diccionario con métricas: Q_table, rewards_history, epsilon_history, steps_history, td_error_history."""
        record_every = 0
        if recorder is not None:
            if n_records is None:
                n_records = self.episodes
            else:
                n_records = max(min( n_records, self.episodes), 1)
            recorder.start()
            record_every = int(self.episodes / n_records)
            if verbose:
                print(f"Recording a training video every: {record_every} episodes")
        
        recorded_times = 0
        door_pos = []
        for episode in tqdm(range(self.episodes)):
            randomized_board = None
            if randomized_boards is not None:
                randomized_board = self.gen_similar_random_board(randomized_boards)
            self.env.reset(new_board = randomized_board)
           
            state = self.env.get_current_state(include_snapshot = True)
            done = False
            episode_reward = 0
            td_error = 0
            
            if recorder is not None:
                record_this_episode = (episode + 1) % record_every == 0
                recorded_times += int(record_this_episode)
                
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = self.step(action)
                td_error += self.update_Q(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                
                if recorder is not None:
                    if record_this_episode:
                        recorder.capture(state["steps"], reward, episode_reward, done)
            #if (episode + 1) % 100 == 0:
            #door_pos.append( self.env.doors[self.env.door_color]["pos"] ) # To keep track if is training in different boards
            avg_td_error = td_error / state["steps"]
            self.td_error_history.append(avg_td_error)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay )
            self.steps_history.append(state["steps"])
            self.rewards_history.append(episode_reward)
            self.epsilon_history.append(self.epsilon)
            
        if recorder is not None:
            recorder.save()
            if verbose:
                print(f"recorded times {recorded_times}")
        
        return { "Q_table": self.Q_table, 
                 "rewards_history": self.rewards_history, 
                 "epsilon_history": self.epsilon_history,
                 "steps_history": self.steps_history,
                 "td_error_history": self.td_error_history,
                 "door_pos": door_pos,
               }

    # ------- evaluate ----------------------------------------------------------------------------

    def evaluate(self, n_episodes: int = 100, render: bool = False, recorder = None, board = None) -> dict:
        """Evalúa la política aprendida con ε=0 (greedy puro) por n_episodes episodios.
        Retorna métricas: tasa de éxito, recompensa media, pasos medios."""
        saved_epsilon = self.epsilon
        self.epsilon = 0.0  # Pure greedy no exploration
    
        successes = 0
        episode_rewards = []
        episode_steps = []
    
        for episode in tqdm(range(n_episodes)):
            self.env.reset(new_board = board)
            if recorder is not None:
                recorder.start()
            state = self.env.get_current_state(include_snapshot=True)
            done = False
            episode_reward = 0.0
    
            while not done:
                action = self.choose_action(state)  # greedy since epsilon=0
                next_state, reward, done, info = self.step(action)
                episode_reward += reward
                state = next_state
                if recorder is not None:
                    recorder.capture(state["steps"], reward, episode_reward, done)
    
                if render:
                    self.env.render()
    
            episode_rewards.append(episode_reward)
            episode_steps.append(state["steps"])
    
            if RewardEvent.EXIT_GOAL in info["events"]:
                successes += 1

        if recorder is not None: # To save only the last episode 
            recorder.save()
    
        self.epsilon = saved_epsilon  # Restore original epsilon
    
        return {
            "success_rate":   successes / n_episodes,
            "mean_reward":    float(np.mean(episode_rewards)),
            "std_reward":     float(np.std(episode_rewards)),
            "mean_steps":     float(np.mean(episode_steps)),
            "std_steps":      float(np.std(episode_steps)),
            "n_episodes":     n_episodes,
            "n_successes":    successes,
        }
        
    # ------- helpers ----------------------------------------------------------------------------

    def get_policy_and_q_values(self) -> tuple[dict,dict]:
        """Deriva y retorna la política greedy actual y q_values a partir de la tabla Q.
        Retorna dict {state_key -> best_action} para todos los estados conocidos."""
        policy = {}
        values = {}
        # Collect all unique state keys seen during training
        known_states = set(state_key for (state_key, _) in self.Q_table.keys())
        for state_key in known_states:
            # Reconstruct a minimal state dict from the encoded key so we can
            # call get_possible_actions which needs i, j, has_key, etc.
            i, j, has_key, has_ball, door_state, door_color, left_room = state_key
            state_dict = {
                "i":          i,
                "j":          j,
                "has_key":    has_key,
                "has_ball":   has_ball,
                "door_state": door_state,
                "door_color": door_color,
                "left_room":  left_room,
                "steps":      0,      # steps is excluded from state key by design
                "snapshot":   None,   # not available without a live env snapshot
            }
            # Get Q-values for all actions recorded for this state
            q_values = {
                action: self.Q_table[(state_key, action)]
                for (sk, action) in self.Q_table.keys()
                if sk == state_key
            }
            if q_values:
                best = max(q_values, key=q_values.get)
                policy[state_key] = best
                values[state_key] = q_values[best]
    
        self.policy = policy
        return policy, values

    def get_recorder(self, path = "videos/train.mp4",  fps = 4, cell_px = 80) -> EpisodeRecorder:
        return EpisodeRecorder(
            path        = path,
            env         = self.env,
            fps         = fps,
            cell_px     = cell_px,
        )
        
    def get_sprite_recorder(self, 
                            path = "videos/sprites_recorder.mp4", 
                            sprite_path = "assets/sprite_sheet.png",
                            sprite_json = "assets/sprite_sheet.json",   # path OR already-parsed dict
                            fps = 4, 
                            cell_px = 99) -> SpriteEpisodeRecorder:
        return SpriteEpisodeRecorder(
            path              = path,
            env               = self.env,
            sprite_sheet_path = sprite_path,
            sprite_json       = sprite_json,
            fps               = fps,
            padding= 140,
            cell_px         = cell_px ,   # native sprite size — scales if you choose another value
        )

    def gen_similar_random_board(self, from_boards: list ) -> list:
        from_board = random.choice(from_boards)
        board_gen = BoardGenerator(from_board)
        return board_gen.generate()

    
    def save(self, path: str) -> None:
        """Serializa y guarda la tabla Q, política y metadatos de entrenamiento en disco.
        Formato: pickle (preserva tipos Python como Action, Color, DoorState, etc.)"""
        import pickle, os
    
        payload = {
            # Core learned data
            "Q_table":          self.Q_table,
            "policy":           self.policy,
            # Hyperparameters (useful for reproducibility)
            "gamma":            self.gamma,
            "alpha":            self.alpha,
            "epsilon":          self.epsilon,
            "epsilon_decay":    self.epsilon_decay,
            "epsilon_min":      self.epsilon_min,
            "initial_epsilon":  self.initial_epsilon,
            "episodes":         self.episodes,
            # Training history
            "rewards_history":  self.rewards_history,
            "steps_history":    self.steps_history,
            "epsilon_history":  self.epsilon_history,
            "td_error_history": self.td_error_history,
        }
    
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        q_size = len(self.Q_table)
        print(f"[save] Saved Q-table ({q_size} entries) -> {path}")
    
    
    def load(self, path: str) -> None:
        """Carga la tabla Q, política e historial de entrenamiento desde disco."""
        import pickle
    
        with open(path, "rb") as f:
            payload = pickle.load(f)
    
        self.Q_table          = payload["Q_table"]
        self.policy           = payload["policy"]
        self.gamma            = payload["gamma"]
        self.alpha            = payload["alpha"]
        self.epsilon          = payload["epsilon"]
        self.epsilon_decay    = payload["epsilon_decay"]
        self.epsilon_min      = payload["epsilon_min"]
        self.initial_epsilon  = payload["initial_epsilon"]
        self.episodes         = payload["episodes"]
        self.rewards_history  = payload["rewards_history"]
        self.steps_history    = payload["steps_history"]
        self.epsilon_history  = payload["epsilon_history"]
        self.td_error_history = payload["td_error_history"]
    
        q_size = len(self.Q_table)
        print(f"[load] Loaded Q-table ({q_size} entries) ← {path}")



    
