from app.simulation.policies.Policy import Policy
from app.data.Scenario import Scenario
from app.simulation.envs.Env import Env
import gymnasium as gym
import numpy as np
import json
import os


class ChildPolicy(Policy):
    """
    Q-LEARNING PRIORITY WEIGHT OPTIMIZER
    
    Based on: "Optimization of Hospital Queue Management Using Priority Queue 
    Algorithm and Reinforcement Learning" (Adhicandra et al., 2024)
    
    IDEA:
    Instead of using PPO to pick a customer directly (hard problem),
    we use Q-Learning to learn the OPTIMAL WEIGHTS for a priority formula:
    
        Pi = w1 * Ki + w2 * Ti + w3 * Ui
    
    Where:
        Ki = Appointment pressure (severity in the paper)
        Ti = Waiting time normalized
        Ui = Abandonment urgency
    
    Q-Learning picks the best (w1, w2, w3) for each SITUATION (state),
    then the priority formula picks the best CUSTOMER automatically.
    
    This is:
    - FASTER to train (minutes vs hours)
    - MORE INTERPRETABLE (you can read the weights)
    - SCIENTIFICALLY VALIDATED (the paper proves it works)
    
    Q-Learning update:
        Q(s,a) ← Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]
    """

    # ---------------------------------------------------------------
    # WEIGHT CONFIGURATIONS (discrete action space for Q-Learning)
    # Each "action" is a set of weights (w1, w2, w3, w4)
    # w1 = appointment weight
    # w2 = waiting time weight
    # w3 = abandonment weight
    # w4 = efficiency weight (service time bonus)
    # ---------------------------------------------------------------
    WEIGHT_CONFIGS = [
        # (w_appt, w_wait, w_abandon, w_efficiency)
        # Balanced
        (0.4, 0.3, 0.2, 0.1),
        (0.3, 0.4, 0.2, 0.1),
        (0.3, 0.3, 0.3, 0.1),

        # Appointment-focused
        (0.6, 0.2, 0.1, 0.1),
        (0.5, 0.3, 0.1, 0.1),
        (0.5, 0.2, 0.2, 0.1),

        # Wait-time-focused (good for Gw score)
        (0.2, 0.6, 0.1, 0.1),
        (0.2, 0.5, 0.2, 0.1),
        (0.1, 0.6, 0.2, 0.1),

        # Abandonment-focused (good for Gs score)
        (0.2, 0.3, 0.4, 0.1),
        (0.2, 0.2, 0.5, 0.1),
        (0.1, 0.3, 0.5, 0.1),

        # Mixed strategies
        (0.4, 0.4, 0.1, 0.1),
        (0.3, 0.5, 0.1, 0.1),
        (0.4, 0.2, 0.3, 0.1),

        # Efficiency-aware
        (0.3, 0.3, 0.2, 0.2),
        (0.2, 0.4, 0.2, 0.2),
        (0.4, 0.3, 0.1, 0.2),
    ]

    # ---------------------------------------------------------------
    # STATE DISCRETIZATION
    # State = (appt_pressure, queue_pressure, abandon_pressure, time_phase)
    # Each dimension is discretized into bins
    # ---------------------------------------------------------------
    # appt_pressure: how many critical appointments right now?
    #   0 = none, 1 = some (1-2), 2 = many (3+)
    APPT_BINS = 3

    # queue_pressure: how full is the queue?
    #   0 = light (<10), 1 = medium (10-25), 2 = heavy (25+)
    QUEUE_BINS = 3

    # abandon_pressure: how many customers close to abandoning?
    #   0 = none, 1 = some, 2 = several
    ABANDON_BINS = 3

    # time_phase: where are we in the day?
    #   0 = morning (<210min), 1 = midday (210-420min), 2 = afternoon (420+min)
    TIME_BINS = 3

    def __init__(self, model_title):
        super().__init__(model_title)

        self.num_actions = len(self.WEIGHT_CONFIGS)
        self.num_states = (
            self.APPT_BINS * self.QUEUE_BINS *
            self.ABANDON_BINS * self.TIME_BINS
        )

        # Q-table: shape (num_states, num_actions)
        self.q_table = np.zeros((self.num_states, self.num_actions))

        # Q-Learning hyperparameters (from the paper: α=0.1, γ=0.9)
        self.alpha = 0.1        # Learning rate
        self.gamma = 0.9        # Discount factor
        self.epsilon = 1.0      # Exploration rate (starts high, decays)
        self.epsilon_min = 0.05 # Minimum exploration
        self.epsilon_decay = 0.995  # Decay per episode #was 0.995 before and gave score 84.6 

        # Rules for critical appointments (always applied regardless of Q-table)
        self.use_critical_rules = True
        self.critical_window = 3  # ±3 min

        # Training state
        self.is_trained = False
        self.model_path = f"models/{model_title}"
        os.makedirs(self.model_path, exist_ok=True)

        # Stats
        self.stats = {
            'rule_decisions': 0,
            'q_decisions': 0,
            'episodes_trained': 0,
        }

    # ---------------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------------

    def _predict(self, obs, info):
        """
        TWO-LEVEL PREDICTION:
        1. Rules: Handle critical appointments (±3 min) immediately
        2. Q-Learning: Choose best weights for priority formula
        """
        action_mask = info.get('action_mask')
        customers = obs['customers']
        context = obs['context']

        # LEVEL 1: Critical appointment rules
        if self.use_critical_rules:
            critical_action = self._check_critical_appointments(customers, action_mask)
            if critical_action is not None:
                self.stats['rule_decisions'] += 1
                return critical_action

        # LEVEL 2: Q-Learning selects best weight configuration
        self.stats['q_decisions'] += 1

        # Get current state
        state = self._get_state(customers, context)

        # Choose action (weight config) - greedy during inference
        weight_idx = int(np.argmax(self.q_table[state]))
        weights = self.WEIGHT_CONFIGS[weight_idx]

        # Apply priority formula with chosen weights
        return self._priority_formula(customers, action_mask, weights)

    def _check_critical_appointments(self, customers, action_mask):
        """
        RULE: Force serve appointments in ±3 min critical window.
        Returns action index or None.
        """
        critical = []
        for i in range(50):
            if customers[i, 8] == 0 or not action_mask[i]:
                continue
            if customers[i, 3] == 1.0 and customers[i, 6] == 1.0:
                # Critical appointment: ±3 min window
                priority = 1000 - customers[i, 4] * 10
                critical.append((i, priority))

        if critical:
            critical.sort(key=lambda x: x[1], reverse=True)
            return critical[0][0]
        return None

    def _priority_formula(self, customers, action_mask, weights):
        """
        PRIORITY FORMULA (from the paper):
        Pi = w1*Ki + w2*Ti + w3*Ui + w4*Efficiency

        Ki = appointment score (higher = more urgent appointment)
        Ti = waiting time normalized
        Ui = abandonment urgency
        Efficiency = short service time bonus
        """
        w_appt, w_wait, w_abandon, w_eff = weights
        scores = np.full(51, -9999.0)

        for i in range(50):
            if customers[i, 8] == 0 or not action_mask[i]:
                continue

            waiting_time = customers[i, 0]      # 0-1
            service_time = customers[i, 2]      # 0-1
            has_appt = customers[i, 3]          # 0 or 1
            appt_urgency = customers[i, 4]      # 0-1 (lower = sooner)
            is_critical = customers[i, 6]       # 1 if ±3 min
            abandon_risk = customers[i, 7]      # 0-1

            # Ki: Appointment score
            if has_appt:
                if is_critical:
                    Ki = 1.0  # Maximum urgency
                else:
                    # Higher score as appointment approaches
                    Ki = max(0, 1.0 - appt_urgency)
            else:
                Ki = 0.0

            # Ti: Waiting time (exponential scaling for long waits)
            Ti = waiting_time ** 1.5

            # Ui: Abandonment urgency
            Ui = abandon_risk

            # Efficiency: Prefer short service for backed-up queue
            Eff = 1.0 - service_time

            # Priority formula
            Pi = w_appt * Ki + w_wait * Ti + w_abandon * Ui + w_eff * Eff
            scores[i] = Pi

        # HOLD gets strong penalty
        scores[50] = -50 if action_mask[50] else -9999

        return int(np.argmax(scores))

    # ---------------------------------------------------------------
    # STATE REPRESENTATION
    # ---------------------------------------------------------------

    def _get_state(self, customers, context):
        """
        Discretize current situation into a Q-table state index.

        State = (appt_pressure, queue_pressure, abandon_pressure, time_phase)
        """
        # 1. Appointment pressure: critical appointments in queue
        critical_appts = sum(
            1 for i in range(50)
            if customers[i, 8] > 0 and
            customers[i, 3] == 1.0 and
            customers[i, 4] < 0.5  # Appointment soon
        )
        if critical_appts == 0:
            appt_bin = 0
        elif critical_appts <= 2:
            appt_bin = 1
        else:
            appt_bin = 2

        # 2. Queue pressure: number of waiting customers
        num_waiting = int(context[1] * 50)  # Denormalize
        if num_waiting < 10:
            queue_bin = 0
        elif num_waiting < 25:
            queue_bin = 1
        else:
            queue_bin = 2

        # 3. Abandonment pressure: customers at risk
        high_risk = sum(
            1 for i in range(50)
            if customers[i, 8] > 0 and customers[i, 7] > 0.7
        )
        if high_risk == 0:
            abandon_bin = 0
        elif high_risk <= 2:
            abandon_bin = 1
        else:
            abandon_bin = 2

        # 4. Time phase: where in the day are we?
        sim_time = context[0] * 630  # Denormalize
        if sim_time < 210:
            time_bin = 0  # Morning
        elif sim_time < 420:
            time_bin = 1  # Midday
        else:
            time_bin = 2  # Afternoon

        # Encode state as single integer
        state = (
            appt_bin * (self.QUEUE_BINS * self.ABANDON_BINS * self.TIME_BINS) +
            queue_bin * (self.ABANDON_BINS * self.TIME_BINS) +
            abandon_bin * self.TIME_BINS +
            time_bin
        )

        return int(state)

    # ---------------------------------------------------------------
    # Q-LEARNING TRAINING
    # ---------------------------------------------------------------

    def learn(self, scenario: Scenario, total_timesteps: int, verbose: int = 1):
        """
        Train Q-Learning by running episodes and updating Q-table.

        Unlike PPO which needs 100k+ steps, Q-Learning converges in
        just a few hundred episodes!
        """
        print(f"\n{'='*60}")
        print(f"Q-LEARNING WEIGHT OPTIMIZER")
        print(f"{'='*60}")
        print(f"Based on: Adhicandra et al. (2024)")
        print(f"Formula: Pi = w1*Ki + w2*Ti + w3*Ui")
        print(f"")
        print(f"States:  {self.num_states} (situation types)")
        print(f"Actions: {self.num_actions} (weight configurations)")
        print(f"α (learning rate): {self.alpha}")
        print(f"γ (discount):      {self.gamma}")
        print(f"Episodes: {total_timesteps}")
        print(f"{'='*60}\n")

        try:
            env = gym.make("Child_Env", mode=Env.MODE.TRAIN, scenario=scenario)
        except Exception as e:
            print(f"⚠ Could not create env: {e}")
            return

        best_reward = -np.inf
        rewards_history = []

        for episode in range(total_timesteps):
            obs, info = env.reset(seed=episode)
            done = False
            total_reward = 0
            prev_state = None
            prev_action = None

            while not done:
                customers = obs['customers']
                context = obs['context']
                action_mask = info.get('action_mask', [True] * 51)

                # Get current state
                state = self._get_state(customers, context)

                # Choose action: ε-greedy
                if np.random.random() < self.epsilon:
                    # Explore: random weight config
                    weight_idx = np.random.randint(self.num_actions)
                else:
                    # Exploit: best known weight config
                    weight_idx = int(np.argmax(self.q_table[state]))

                weights = self.WEIGHT_CONFIGS[weight_idx]

                # Check critical appointments first (rules)
                if self.use_critical_rules:
                    critical = self._check_critical_appointments(customers, action_mask)
                    if critical is not None:
                        env_action = critical
                    else:
                        env_action = self._priority_formula(
                            customers, action_mask, weights
                        )
                else:
                    env_action = self._priority_formula(
                        customers, action_mask, weights
                    )

                # Step environment
                next_obs, reward, terminated, truncated, next_info = env.step(env_action)
                done = terminated or truncated
                total_reward += reward

                # Q-Learning update (from the paper)
                if prev_state is not None:
                    next_state = self._get_state(
                        next_obs['customers'], next_obs['context']
                    )
                    best_next = np.max(self.q_table[next_state])

                    # Q(s,a) ← Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)]
                    self.q_table[prev_state, prev_action] += self.alpha * (
                        reward +
                        self.gamma * best_next -
                        self.q_table[prev_state, prev_action]
                    )

                prev_state = state
                prev_action = weight_idx
                obs = next_obs
                info = next_info

            # Decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            self.stats['episodes_trained'] += 1

            # Track best model
            if total_reward > best_reward:
                best_reward = total_reward

            # Logging
            if verbose and (episode + 1) % 50 == 0:
                recent_avg = np.mean(rewards_history[-50:])
                print(f"Episode {episode+1:4d}/{total_timesteps} | "
                      f"Avg Reward: {recent_avg:8.1f} | "
                      f"Best: {best_reward:8.1f} | "
                      f"ε: {self.epsilon:.3f}")

        env.close()
        self.is_trained = True

        # Save Q-table
        self.save_model()

        print(f"\n{'='*60}")
        print(f"✓ Q-Learning training complete!")
        print(f"✓ Episodes trained: {total_timesteps}")
        print(f"✓ Final ε: {self.epsilon:.3f}")
        print(f"✓ Best reward: {best_reward:.1f}")
        print(f"\nLearned weight preferences per state:")
        self._print_learned_weights()
        print(f"{'='*60}\n")

    def _print_learned_weights(self):
        """Show which weight configs Q-Learning prefers for each situation."""
        state_names = []
        for a in range(self.APPT_BINS):
            for q in range(self.QUEUE_BINS):
                for ab in range(self.ABANDON_BINS):
                    for t in range(self.TIME_BINS):
                        appt_label = ['No appts', 'Some appts', 'Many appts'][a]
                        queue_label = ['Light queue', 'Medium queue', 'Heavy queue'][q]
                        time_label = ['Morning', 'Midday', 'Afternoon'][t]
                        state_idx = (
                            a * (self.QUEUE_BINS * self.ABANDON_BINS * self.TIME_BINS) +
                            q * (self.ABANDON_BINS * self.TIME_BINS) +
                            ab * self.TIME_BINS + t
                        )
                        best_action = int(np.argmax(self.q_table[state_idx]))
                        w = self.WEIGHT_CONFIGS[best_action]
                        # Only print non-zero states
                        if np.max(self.q_table[state_idx]) > 0:
                            print(f"  [{appt_label}, {queue_label}, {time_label}]"
                                  f" → w_appt={w[0]}, w_wait={w[1]},"
                                  f" w_abandon={w[2]}, w_eff={w[3]}")

    # ---------------------------------------------------------------
    # SAVE / LOAD
    # ---------------------------------------------------------------

    def save_model(self, path: str = None):
        if path is None:
            path = f"{self.model_path}/q_table.npy"
        np.save(path, self.q_table)
        # Save metadata
        meta = {
            'epsilon': float(self.epsilon),
            'episodes_trained': self.stats['episodes_trained'],
            'num_states': self.num_states,
            'num_actions': self.num_actions,
        }
        with open(f"{self.model_path}/q_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"✓ Q-table saved to: {path}")

    def load_model(self, path: str = None):
        if path is None:
            path = f"{self.model_path}/q_table.npy"
        try:
            self.q_table = np.load(path)
            self.epsilon = self.epsilon_min  # Greedy during inference
            self.is_trained = True
            print(f"✓ Q-table loaded from: {path}")
        except FileNotFoundError:
            print(f"⚠ No Q-table found at {path}, using untrained weights")

    def print_stats(self):
        total = self.stats['rule_decisions'] + self.stats['q_decisions']
        if total > 0:
            print(f"\n{'='*60}")
            print(f"Q-LEARNING POLICY STATISTICS")
            print(f"{'='*60}")
            print(f"Rule-based (critical appts): {self.stats['rule_decisions']} "
                  f"({100*self.stats['rule_decisions']/total:.1f}%)")
            print(f"Q-Learning decisions:        {self.stats['q_decisions']} "
                  f"({100*self.stats['q_decisions']/total:.1f}%)")
            print(f"Episodes trained:            {self.stats['episodes_trained']}")
            print(f"{'='*60}")