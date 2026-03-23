from app.simulation.envs.Env import Env
from app.domain.Customer import Customer
import gymnasium as gym
import numpy as np

N = 50  # Customer pool size

class ChildEnv(Env):
    """
    Environment for Hospital Queue Management.
    
    Action Space: Discrete(51) - Select from 50 customers + HOLD
    Observation: 50×9 customer features + 8 context features (458 total)
    Reward: Aligned with evaluation (40% waiting, 40% appointments, 20% throughput)
    """
    
    def _get_action_space(self):
        return gym.spaces.Discrete(N + 1)
    
    def _get_observation_space(self):
        return gym.spaces.Dict({
            "customers": gym.spaces.Box(low=0.0, high=1.0, shape=(N, 9), dtype=np.float32),
            "context": gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        })
    
    def _get_obs(self):
        """Build observation with customer features and global context."""
        customer_ids = sorted(self.customer_waiting.keys(), 
                            key=lambda cid: self.customer_waiting[cid].arrival_time)
        
        customer_features = np.zeros((N, 9), dtype=np.float32)
        current_server = self.current_working_server
        sim_time = self.system_time
        
        # Fill customer features
        for i, customer_id in enumerate(customer_ids[:N]):
            customer = self.customer_waiting[customer_id]
            
            # Waiting time (normalized by 60 min)
            waiting_time = sim_time - customer.arrival_time
            customer_features[i, 0] = min(waiting_time / 60.0, 1.0)
            
            # Task compatibility
            can_handle = current_server.avg_service_time[customer.task] > 0
            customer_features[i, 1] = 1.0 if can_handle else 0.0
            
            # Service time (normalized by 30 min)
            if can_handle:
                service_time = current_server.avg_service_time[customer.task]
                customer_features[i, 2] = min(service_time / 30.0, 1.0)
            else:
                customer_features[i, 2] = 1.0
            
            # Appointment features
            has_appointment = customer_id in self.appointments
            customer_features[i, 3] = 1.0 if has_appointment else 0.0
            
            if has_appointment:
                appt = self.appointments[customer_id]
                time_to_appt = appt.time - sim_time
                
                # Appointment urgency
                customer_features[i, 4] = np.clip((time_to_appt + 30) / 60.0, 0.0, 1.0)
                
                # Delay if late
                if sim_time > appt.time:
                    delay = sim_time - appt.time
                    customer_features[i, 5] = min(delay / 30.0, 1.0)
                else:
                    customer_features[i, 5] = 0.0
                
                # Critical window (±3 min)
                is_critical = abs(sim_time - appt.time) <= 3
                customer_features[i, 6] = 1.0 if is_critical else 0.0
            else:
                customer_features[i, 4] = 0.0
                customer_features[i, 5] = 0.0
                customer_features[i, 6] = 0.0
            
            # Abandonment risk
            if customer.abandonment_time is not None:
                time_until_abandon = customer.abandonment_time - sim_time
                if time_until_abandon > 0:
                    customer_features[i, 7] = 1.0 - min(time_until_abandon / 60.0, 1.0)
                else:
                    customer_features[i, 7] = 1.0
            else:
                customer_features[i, 7] = 0.0
            
            # Customer exists (padding mask)
            customer_features[i, 8] = 1.0
        
        # Global context features
        context = np.zeros(8, dtype=np.float32)
        
        context[0] = min(sim_time / 630.0, 1.0)  # Simulation time
        context[1] = min(len(self.customer_waiting) / N, 1.0)  # Queue size
        context[2] = current_server.id / max(self.c - 1, 1)  # Server ID
        
        # Server busy ratio
        busy_count = sum(1 for sid in range(self.c) if sid in self.current_server_activity)
        context[3] = busy_count / max(self.c, 1)
        
        # Average waiting time
        if len(self.customer_waiting) > 0:
            avg_wait = np.mean([sim_time - c.arrival_time for c in self.customer_waiting.values()])
            context[4] = min(avg_wait / 60.0, 1.0)
        else:
            context[4] = 0.0
        
        # Appointment pressure (upcoming in next 30 min)
        if len(self.appointments) > 0:
            upcoming = sum(1 for a in self.appointments.values() if sim_time <= a.time <= sim_time + 30)
            context[5] = upcoming / max(len(self.appointments), 1)
        else:
            context[5] = 0.0
        
        context[6] = max(0.0, (self.max_sim_time - sim_time) / self.max_sim_time)  # Time remaining
        
        # Served ratio
        total_arrived = sum(1 for c in self.customers_arrival.values() if c.arrival_time <= sim_time)
        if total_arrived > 0:
            context[7] = self.served_clients / total_arrived
        else:
            context[7] = 0.0
        
        return {"customers": customer_features, "context": context}
    
    def _get_customer_from_action(self, action) -> Customer:
        """Convert action index to customer object."""
        if action == N:
            return None
        
        customer_ids = sorted(self.customer_waiting.keys(),
                            key=lambda cid: self.customer_waiting[cid].arrival_time)
        
        if action >= len(customer_ids):
            return None
        
        customer_id = customer_ids[action]
        
        if customer_id not in self.customer_waiting:
            return None
        
        customer = self.customer_waiting[customer_id]
        
        # Verify server can handle task
        if self.current_working_server.avg_service_time[customer.task] <= 0:
            return None
        
        return customer
    
    def _get_invalid_action_reward(self) -> float:
        return -100.0
    
    def _get_valid_reward(self, customer: Customer) -> float:
        """
        Calculate reward aligned with evaluation metrics.
        40% waiting, 40% appointments, 20% throughput.
        """
        waiting_time = self.system_time - customer.arrival_time
        reward = 0.0
        
        # Waiting time score (0-100)
        if waiting_time <= 60:
            wait_score = 100 * (1 - waiting_time / 60.0)
        else:
            wait_score = 0
        
        # Appointment score (0-100)
        appt_score = 0
        is_appointment = customer.id in self.appointments
        
        if is_appointment:
            appt = self.appointments[customer.id]
            delay = self.system_time - appt.time
            
            if abs(delay) <= 3:
                appt_score = 100
            elif -60 < delay < -3:
                appt_score = 100 * (1 + (delay + 3) / 57.0)
            elif 3 < delay < 30:
                appt_score = 100 * (1 - (delay - 3) / 27.0)
            else:
                appt_score = 0
        
        throughput_score = 100
        
        # Weighted combination
        if is_appointment:
            reward = 0.4 * appt_score + 0.2 * throughput_score
        else:
            reward = 0.4 * wait_score + 0.2 * throughput_score
        
        # Bonus: prevent abandonment
        if customer.abandonment_time is not None:
            time_until_abandon = customer.abandonment_time - self.system_time
            if 0 < time_until_abandon < 5:
                reward += 20
        
        # Bonus: perfect appointment timing
        if is_appointment and abs(delay) <= 3:
            reward += 10
        
        return reward
    
    def action_masks(self):
        """Create mask to disable invalid actions."""
        num_waiting = len(self.customer_waiting)
        
        if num_waiting == 0:
            return [False] * N + [True]
        
        customer_ids = sorted(self.customer_waiting.keys(),
                            key=lambda cid: self.customer_waiting[cid].arrival_time)
        
        mask = [False] * (N + 1)
        
        for i, customer_id in enumerate(customer_ids[:N]):
            customer = self.customer_waiting[customer_id]
            can_handle = self.current_working_server.avg_service_time[customer.task] > 0
            mask[i] = can_handle
        
        mask[N] = True  # HOLD always valid
        
        return mask
    
    def _get_hold_action_number(self):
        return N