import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Actions import Action
from EnvironmentProperties import CellType
from EnvironmentProperties import Color 
from EnvironmentProperties import DoorState 

class QLearningPlotter:
    """
    Plotting utilities for a trained QLearningAgent.

    Usage
    -----
    plotter = QLearningPlotter(agent)
    plotter.plot_heatmap()
    plotter.plot_training()
    plotter.plot_visit_frequency()
    plotter.plot_action_uncertainty()
    plotter.plot_per_action_heatmap()
    plotter.plot_policy_consistency()
    """

    def __init__(self, agent):
        self.agent = agent

    # ─────────────────────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────────────────────

    def _base_grid(self):
        env = self.agent.env
        nrows, ncols = env.nrows, env.ncols
        fig, ax = plt.subplots(figsize=(ncols * 1.2, nrows * 1.2))
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        return fig, ax

    def _add_grid(self, ax):
        env = self.agent.env
        for x in range(env.ncols + 1): ax.axvline(x, color="black", linewidth=0.8)
        for y in range(env.nrows + 1): ax.axhline(y, color="black", linewidth=0.8)

    def _add_labels(self, ax):
        env   = self.agent.env
        nrows = env.nrows
        si, sj = env.initial_state
        ax.text(sj+0.5, nrows-si-0.15, "S", ha="center", va="top", fontsize=8, color="blue")
        gi, gj = env.end_state_pos
        ax.text(gj+0.5, nrows-gi-0.15, "G", ha="center", va="top", fontsize=8, color="green")

    def _add_colorbar(self, ax, cmap, vmin, vmax, label):
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax)),
                     ax=ax, fraction=0.03, label=label)

    def _draw_walls(self, ax, nrows, ncols):
        """Draws wall patches, returns a boolean mask."""
        env       = self.agent.env
        wall_mask = np.zeros((nrows, ncols), dtype=bool)
        for i in range(nrows):
            for j in range(ncols):
                if env.grid[i][j] == CellType.WALL.code:
                    y = nrows - i - 1
                    ax.add_patch(patches.Rectangle((j, y), 1, 1, color="#444444"))
                    wall_mask[i][j] = True
        return wall_mask

    def _q_table_iter(self):
        """Yields (i, j, left_room, state_key, action, q_val) filtered by left_room consistency."""
        env      = self.agent.env
        door_col = env.doors[env.door_color]["pos"][1]
        nrows, ncols = env.nrows, env.ncols
        for (state_key, action), q_val in self.agent.Q_table.items():
            i, j, _, _, _, _, left_room = state_key
            if left_room != (j > door_col): continue
            if not (0 <= i < nrows and 0 <= j < ncols): continue
            yield i, j, left_room, state_key, action, q_val

    ARROW = {
        Action.UP: "^", Action.DOWN: "v",
        Action.LEFT: "<", Action.RIGHT: ">",
        Action.PICK_UP_ITEM: "P", Action.DROP_ITEM: "D",
        Action.USE_ITEM: "U", Action.EXIT: "X",
    }


    # ─────────────────────────────────────────────────────────────
    # 1. Q -Table Data Fraame
    # ─────────────────────────────────────────────────────────────
    

    def to_dataframe(self):
        """Returns the Q-table as a tidy pandas DataFrame, one row per (state, action)."""
        import pandas as pd
    
        rows = []
        for i, j, left_room, state_key, action, q_val in self._q_table_iter():
            _, _, has_key, has_ball, door_state, door_color, _ = state_key
            rows.append({
                "i":          i,
                "j":          j,
                "has_key":    has_key,
                "has_ball":   has_ball,
                "door_state": door_state.name,
                "door_color": door_color.name,
                "left_room":  left_room,
                "action":     self.ARROW.get(action, action.name),
                "q_value":    round(q_val, 4),
            })
    
        df = pd.DataFrame(rows).sort_values(
            ["i", "j", "door_state", "has_key", "has_ball", "action"]
        ).reset_index(drop=True)
    
        return df

    def to_best_policy_dataframe(self):
        """
        Usage:
        
        plotter = QLearningPlotter(agent)

        # Full Q-table
        plotter.to_dataframe()
        
        # One row per cell — best action + V*(s) value
        plotter.to_best_policy_dataframe()
        
        # Plain dict for quick lookup or passing to other methods
        plotter.to_policy_dict()
        
        # Useful slices on best_df
        best_df = plotter.to_best_policy_dataframe()
        
        # Which cells the agent is most confident about
        best_df.sort_values("q_value", ascending=False).head(10)
        
        # Cells with negative best Q — agent expects to lose from here
        best_df[best_df["q_value"] < 0]
        
        # Action distribution across the grid
        best_df["action"].value_counts()
        
        Returns a DataFrame with one row per (i,j) cell — the state with
        the highest Q value across all contexts, plus the corresponding best action.
        Equivalent to V*(s) + policy per grid position.
        """
        import pandas as pd
    
        df  = self.to_dataframe()
        env = self.agent.env
        door_col   = env.doors[env.door_color]["pos"][1]
        nrows, ncols = env.nrows, env.ncols
    
        filtered = df[
            (df["left_room"] == (df["j"] > door_col)) &
            (df["i"].between(0, nrows - 1)) &
            (df["j"].between(0, ncols - 1))
        ]
    
        idx     = filtered.groupby(["i", "j"])["q_value"].idxmax()
        best_df = filtered.loc[idx].reset_index(drop=True)
    
        return best_df
    
    
    def to_policy_dict(self) -> dict:
        """
        Returns the policy as a plain dict {(i,j): action_symbol}
        derived from to_best_policy_dataframe — consistent with plot_heatmap.
        """
        best_df = self.to_best_policy_dataframe()
        return {
            (row.i, row.j): row.action
            for row in best_df.itertuples()
        }

    # ─────────────────────────────────────────────────────────────
    # 1. V*(s) heatmap + policy arrows
    # ─────────────────────────────────────────────────────────────

    
    def plot_heatmap(self) -> None:
        """ Plots Q table V*(s) heatmap + policy arrows over the grid """
        env        = self.agent.env
        nrows      = env.nrows
        ncols      = env.ncols
        door_color = env.door_color
        door_col   = env.doors[door_color]["pos"][1]

        ARROW = {
            Action.UP: "^", Action.DOWN: "v",
            Action.LEFT: "<", Action.RIGHT: ">",
            Action.PICK_UP_ITEM: "P", Action.DROP_ITEM: "D",
            Action.USE_ITEM: "U", Action.EXIT: "X",
        }

        values = np.full((nrows, ncols), np.nan)
        policy = {}

        full_policy, q_values = self.agent.get_policy_and_q_values()
        for state_key, best_action in full_policy.items():
            i, j, _, _, _, _, left_room = state_key
            if left_room != (j > door_col):
                continue
            if not (0 <= i < nrows and 0 <= j < ncols):
                continue
            if np.isnan(values[i][j]) or q_values[state_key] > values[i][j]:
                values[i][j]  = q_values[state_key]
                policy[(i, j)] = best_action

        fig, ax = plt.subplots(figsize=(ncols * 1.2, nrows * 1.2))
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)

        vmin, vmax = np.nanmin(values), np.nanmax(values)
        cmap = plt.get_cmap("RdYlGn")

        for i in range(nrows):
            for j in range(ncols):
                y   = nrows - i - 1
                val = values[i][j]

                if env.grid[i][j] == CellType.WALL.code:
                    ax.add_patch(patches.Rectangle((j, y), 1, 1, color="#444444"))
                    continue

                color = cmap((val - vmin) / (vmax - vmin)) if not np.isnan(val) else "#222222"
                ax.add_patch(patches.Rectangle((j, y), 1, 1, color=color))

                if not np.isnan(val):
                    ax.text(j+0.5, y+0.65, f"{val:.2f}",
                            ha="center", va="center", fontsize=7, fontweight="bold")
                    ax.text(j+0.5, y+0.30, ARROW.get(policy.get((i, j)), ""),
                            ha="center", va="center", fontsize=9)

        si, sj = env.initial_state
        ax.text(sj+0.5, nrows-si-0.15, "S", ha="center", va="top", fontsize=8, color="blue")
        gi, gj = env.end_state_pos
        ax.text(gj+0.5, nrows-gi-0.15, "G", ha="center", va="top", fontsize=8, color="green")

        for x in range(ncols + 1):
            ax.axvline(x, color="black", linewidth=0.8)
        for y in range(nrows + 1):
            ax.axhline(y, color="black", linewidth=0.8)

        plt.colorbar(plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin, vmax)),
            ax=ax, fraction=0.03, label="max Q(s,a)")

        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.set_xticklabels(range(ncols))
        ax.set_yticklabels(range(nrows - 1, -1, -1))
        ax.set_title("Q-Table Heatmap — V*(s) + Best Action")
        plt.tight_layout()
        plt.show()

    def plot_heatmap2(self) -> None:
        """V*(s) = max_a Q(s,a) per cell + best action arrow."""
        ARROW = {
            Action.UP: "^", Action.DOWN: "v",
            Action.LEFT: "<", Action.RIGHT: ">",
            Action.PICK_UP_ITEM: "P", Action.DROP_ITEM: "D",
            Action.USE_ITEM: "U", Action.EXIT: "X",
        }
        env   = self.agent.env
        nrows, ncols = env.nrows, env.ncols
        values = np.full((nrows, ncols), np.nan)
        arrows = {}

        for i, j, _, state_key, action, q_val in self._q_table_iter():
            if np.isnan(values[i][j]) or q_val > values[i][j]:
                values[i][j] = q_val
                arrows[(i,j)] = self.agent.policy.get(state_key)

        fig, ax = self._base_grid()
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        cmap = plt.get_cmap("RdYlGn")
        wall_mask = self._draw_walls(ax, nrows, ncols)

        for i in range(nrows):
            for j in range(ncols):
                if wall_mask[i][j]: continue
                y   = nrows - i - 1
                val = values[i][j]
                color = cmap((val-vmin)/(vmax-vmin)) if not np.isnan(val) else "#222222"
                ax.add_patch(patches.Rectangle((j,y), 1, 1, color=color))
                if not np.isnan(val):
                    ax.text(j+0.5, y+0.65, f"{val:.2f}",
                            ha="center", va="center", fontsize=7, fontweight="bold")
                    ax.text(j+0.5, y+0.30, self.ARROW.get(arrows.get((i,j)), ""),
                            ha="center", va="center", fontsize=9)

        self._add_labels(ax)
        self._add_grid(ax)
        self._add_colorbar(ax, cmap, vmin, vmax, "max Q(s,a)")
        ax.set_xticks(range(ncols)); ax.set_xticklabels(range(ncols))
        ax.set_yticks(range(nrows)); ax.set_yticklabels(range(nrows-1, -1, -1))
        ax.set_title("Q-Table Heatmap — V*(s) + Best Action")
        plt.tight_layout(); plt.show()

    # ─────────────────────────────────────────────────────────────
    # 2. Visit frequency
    # ─────────────────────────────────────────────────────────────

    def plot_visit_frequency(self) -> None:
        """Q-entries per cell — proxy for how often the agent visited each state."""
        env   = self.agent.env
        nrows, ncols = env.nrows, env.ncols
        counts = np.zeros((nrows, ncols))

        for i, j, *_ in self._q_table_iter():
            counts[i][j] += 1

        fig, ax = self._base_grid()
        vmin, vmax = counts.min(), counts.max()
        cmap = plt.get_cmap("YlOrRd")
        wall_mask = self._draw_walls(ax, nrows, ncols)

        for i in range(nrows):
            for j in range(ncols):
                if wall_mask[i][j]: continue
                y     = nrows - i - 1
                color = cmap((counts[i][j]-vmin)/(vmax-vmin)) if vmax > 0 else "#222222"
                ax.add_patch(patches.Rectangle((j,y), 1, 1, color=color))
                ax.text(j+0.5, y+0.5, f"{int(counts[i][j])}",
                        ha="center", va="center", fontsize=7, fontweight="bold")

        self._add_labels(ax)
        self._add_grid(ax)
        self._add_colorbar(ax, cmap, vmin, vmax, "Q-entries per cell")
        ax.set_title("State Visit Frequency")
        plt.tight_layout(); plt.show()

    # ─────────────────────────────────────────────────────────────
    # 3. Action uncertainty (Q spread)
    # ─────────────────────────────────────────────────────────────

    def plot_action_uncertainty(self) -> None:
        """max Q - min Q per cell — high spread means the agent is still uncertain."""
        env   = self.agent.env
        nrows, ncols = env.nrows, env.ncols
        cell_q = {}

        for i, j, _, state_key, action, q_val in self._q_table_iter():
            cell_q.setdefault((i,j), []).append(q_val)

        spread = np.full((nrows, ncols), np.nan)
        for (i,j), vals in cell_q.items():
            spread[i][j] = max(vals) - min(vals)

        fig, ax = self._base_grid()
        vmin, vmax = np.nanmin(spread), np.nanmax(spread)
        cmap = plt.get_cmap("RdYlGn_r")   # red = high uncertainty
        wall_mask = self._draw_walls(ax, nrows, ncols)

        for i in range(nrows):
            for j in range(ncols):
                if wall_mask[i][j]: continue
                y   = nrows - i - 1
                val = spread[i][j]
                color = cmap((val-vmin)/(vmax-vmin)) if not np.isnan(val) else "#222222"
                ax.add_patch(patches.Rectangle((j,y), 1, 1, color=color))
                if not np.isnan(val):
                    ax.text(j+0.5, y+0.5, f"{val:.2f}",
                            ha="center", va="center", fontsize=7, fontweight="bold")

        self._add_labels(ax)
        self._add_grid(ax)
        self._add_colorbar(ax, cmap, vmin, vmax, "max Q - min Q")
        ax.set_title("Action Uncertainty per Cell (Q spread)")
        plt.tight_layout(); plt.show()

    # ─────────────────────────────────────────────────────────────
    # 4. Per-action Q heatmap
    # ─────────────────────────────────────────────────────────────

    def plot_per_action_heatmap(self) -> None:
        """One subplot per action showing Q(s,a) across the grid."""
        env   = self.agent.env
        nrows, ncols = env.nrows, env.ncols
        actions = list(Action)
        q_grids = {a: np.full((nrows, ncols), np.nan) for a in actions}

        for i, j, _, state_key, action, q_val in self._q_table_iter():
            if np.isnan(q_grids[action][i][j]) or q_val > q_grids[action][i][j]:
                q_grids[action][i][j] = q_val

        ncols_plot = 4
        nrows_plot = (len(actions) + ncols_plot - 1) // ncols_plot
        fig, axes  = plt.subplots(nrows_plot, ncols_plot,
                                   figsize=(ncols * ncols_plot * 0.9, nrows * nrows_plot * 0.9))
        axes = axes.flatten()
        cmap = plt.get_cmap("RdYlGn")

        for idx, action in enumerate(actions):
            ax   = axes[idx]
            grid = q_grids[action]
            valid = grid[~np.isnan(grid)]
            vmin  = valid.min() if len(valid) else 0
            vmax  = valid.max() if len(valid) else 1
            ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)

            for i in range(nrows):
                for j in range(ncols):
                    y   = nrows - i - 1
                    val = grid[i][j]
                    if env.grid[i][j] == CellType.WALL.code:
                        ax.add_patch(patches.Rectangle((j,y),1,1,color="#444444")); continue
                    color = cmap((val-vmin)/(vmax-vmin+1e-9)) if not np.isnan(val) else "#222222"
                    ax.add_patch(patches.Rectangle((j,y), 1, 1, color=color))
                    if not np.isnan(val):
                        ax.text(j+0.5, y+0.5, f"{val:.1f}",
                                ha="center", va="center", fontsize=6)
            for x in range(ncols+1): ax.axvline(x, color="black", linewidth=0.5)
            for y in range(nrows+1): ax.axhline(y, color="black", linewidth=0.5)
            ax.set_title(f"{self.ARROW.get(action,'')} {action.name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        for idx in range(len(actions), len(axes)):
            axes[idx].axis("off")

        plt.suptitle("Q(s,a) per Action", fontsize=13, fontweight="bold")
        plt.tight_layout(); plt.show()

    # ─────────────────────────────────────────────────────────────
    # 5. Policy consistency
    # ─────────────────────────────────────────────────────────────

    def plot_policy_consistency(self) -> None:
        """
        Fraction of state contexts that agree on the best action per cell.
        1.0 = agent always picks the same action regardless of inventory/door state.
        """
        from collections import Counter
        env   = self.agent.env
        nrows, ncols = env.nrows, env.ncols
        cell_actions = {}

        for i, j, _, state_key, action, q_val in self._q_table_iter():
            best = self.agent.policy.get(state_key)
            if best is not None:
                cell_actions.setdefault((i,j), []).append(best)

        consistency = np.full((nrows, ncols), np.nan)
        dominant    = {}
        for (i,j), action_list in cell_actions.items():
            counts = Counter(action_list)
            best_action, best_count = counts.most_common(1)[0]
            consistency[i][j] = best_count / len(action_list)
            dominant[(i,j)]   = best_action

        fig, ax = self._base_grid()
        cmap = plt.get_cmap("RdYlGn")
        wall_mask = self._draw_walls(ax, nrows, ncols)

        for i in range(nrows):
            for j in range(ncols):
                if wall_mask[i][j]: continue
                y   = nrows - i - 1
                val = consistency[i][j]
                color = cmap(val) if not np.isnan(val) else "#222222"
                ax.add_patch(patches.Rectangle((j,y), 1, 1, color=color))
                if not np.isnan(val):
                    ax.text(j+0.5, y+0.65, f"{val:.0%}",
                            ha="center", va="center", fontsize=6, fontweight="bold")
                    ax.text(j+0.5, y+0.30, self.ARROW.get(dominant.get((i,j)), ""),
                            ha="center", va="center", fontsize=9)

        self._add_labels(ax)
        self._add_grid(ax)
        self._add_colorbar(ax, cmap, 0, 1, "Action agreement (1 = always same)")
        ax.set_title("Policy Consistency across State Contexts")
        plt.tight_layout(); plt.show()

    # ─────────────────────────────────────────────────────────────
    # 6. Policy  
    # ─────────────────────────────────────────────────────────────

    def plot_policy(self) -> None:
        """Policy arrows + V*(s) values per cell, styled after the reference plot_policy function."""
        env          = self.agent.env
        nrows, ncols = env.nrows, env.ncols
        door_col     = env.doors[env.door_color]["pos"][1]
    
        # ── Build values and policy dicts from Q-table ────────────────────────
        values = np.full((nrows, ncols), np.nan)
        policy = {}
    
        full_policy, q_values = self.agent.get_policy_and_q_values()
        for state_key, best_action in full_policy.items():
            i, j, _, _, _, _, left_room = state_key
            if left_room != (j > door_col):          continue
            if not (0 <= i < nrows and 0 <= j < ncols): continue
            if np.isnan(values[i][j]) or q_values[state_key] > values[i][j]:
                values[i][j]  = q_values[state_key]
                policy[(i, j)] = best_action
    
        # ── Arrow deltas per action ───────────────────────────────────────────
        DELTAS = {
            Action.UP:    (0,   0.25),
            Action.DOWN:  (0,  -0.25),
            Action.LEFT:  (-0.25, 0),
            Action.RIGHT: ( 0.25, 0),
        }
        SYMBOLS = {
            Action.PICK_UP_ITEM: "PICK",
            Action.DROP_ITEM:    "DROP",
            Action.USE_ITEM:     "USE",
            Action.EXIT:         "EXIT",
        }
    
        # ── Plot ─────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(ncols * 1.2, nrows * 1.2))
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.set_aspect("equal")
    
        si, sj = env.initial_state
        gi, gj = env.end_state_pos
    
        for i in range(nrows):
            for j in range(ncols):
                y = nrows - i - 1
    
                # Cell border
                ax.add_patch(patches.Rectangle((j, y), 1, 1, fill=False, edgecolor="black", linewidth=0.8))
    
                # Wall
                if env.grid[i][j] == CellType.WALL.code:
                    ax.add_patch(patches.Rectangle((j, y), 1, 1, color="black"))
                    continue
    
                # Value (top of cell)
                val = values[i][j]
                if not np.isnan(val):
                    ax.text(j + 0.5, y + 0.78, f"{val:.2f}",
                            ha="center", va="center", fontsize=7, fontweight="bold")
    
                # Policy arrow or symbol (center of cell)
                action = policy.get((i, j))
                if action in DELTAS:
                    dx, dy = DELTAS[action]
                    ax.arrow(j + 0.5, y + 0.42, dx, dy,
                             head_width=0.12, head_length=0.10, fc="black", ec="black")
                elif action in SYMBOLS:
                    ax.text(j + 0.5, y + 0.42, SYMBOLS[action],
                            ha="center", va="center", fontsize=10)
    
                # Special cell labels (bottom of cell)
                if (i, j) == (si, sj):
                    ax.text(j + 0.5, y + 0.18, "S",
                            ha="center", va="center", fontsize=9, color="blue", fontweight="bold")
                elif (i, j) == (gi, gj):
                    ax.text(j + 0.5, y + 0.18, "G",
                            ha="center", va="center", fontsize=9, color="green", fontweight="bold")
    
        ax.set_xticks(range(ncols)); ax.set_xticklabels([])
        ax.set_yticks(range(nrows)); ax.set_yticklabels([])
        ax.grid(True, linewidth=0.5)
        plt.title("Policy and Value Function")
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────
    # 6. Training learnign curves  
    # ─────────────────────────────────────────────────────────────


    def plot_training(self) -> None:
        """Grafica rewards_history, steps_history, epsilon_history y td_error_history."""
        agent = self.agent

        if not agent.rewards_history:
            print("plot_training: no training data found. Run train() first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Q-Learning Training Diagnostics", fontsize=15, fontweight="bold")

        episodes = range(1, len(agent.rewards_history) + 1)
        window   = min(50, len(agent.rewards_history))

        ax = axes[0, 0]
        ax.plot(episodes, agent.rewards_history, color="#2196F3", linewidth=0.8, alpha=0.6)
        smoothed = np.convolve(agent.rewards_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(agent.rewards_history) + 1), smoothed,
                color="#F44336", linewidth=1.8, label=f"Rolling mean ({window})")
        ax.set_title("Total Reward per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(episodes, agent.steps_history, color="#4CAF50", linewidth=0.8, alpha=0.6)
        smoothed_steps = np.convolve(agent.steps_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(agent.steps_history) + 1), smoothed_steps,
                color="#FF9800", linewidth=1.8, label=f"Rolling mean ({window})")
        ax.set_title("Steps per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(episodes, agent.epsilon_history, color="#9C27B0", linewidth=1.5)
        ax.set_title("Epsilon Decay (Exploration Rate)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_ylim(0, max(agent.epsilon_history) * 1.05)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if agent.td_error_history:
            td        = agent.td_error_history
            # step_size = max(1, len(td) // 5000)
            step_size = max(1, 1)
            td_sampled = td[::step_size]
            ax.plot(range(0, len(td), step_size), td_sampled,
                    color="#607D8B", linewidth=0.5, alpha=0.5)
            w2        = min(200, len(td_sampled))
            td_smooth = np.convolve(td_sampled, np.ones(w2) / w2, mode="valid")
            ax.plot(range(w2 - 1, len(td_sampled)), td_smooth,
                    color="#E91E63", linewidth=1.8, label=f"Rolling mean ({w2})")
            ax.legend(fontsize=8)
        ax.set_title("|TD Error| per Step")
        ax.set_xlabel("Training Step (sampled)")
        ax.set_ylabel("|TD Error|")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()