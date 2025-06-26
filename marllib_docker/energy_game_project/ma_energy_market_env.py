from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, from_parallel
from typing import Optional
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime


class parallel_env(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "energy_market_v1"}

    def __init__(self, 
                max_gen_power, min_gen_power, 
                max_con_price, min_con_price, 
                ind_max_gen_power, ind_min_gen_power, 
                ind_max_con_price, ind_min_con_price,
                max_steps,
                num_generators=1, num_consumers=1):
        """
        Initialize the Energy Market environment.
        
        Args:
            max_gen_power: Maximum TOTAL generator power output (collective bound)
            min_gen_power: Minimum TOTAL generator power output (collective bound)
            max_con_price: Maximum TOTAL consumer price (collective bound)
            min_con_price: Minimum TOTAL consumer price (collective bound)
            profit_threshold: Profit threshold for winning condition
            max_steps: Maximum number of steps before episode ends
            num_generators: Number of generator agents
            num_consumers: Number of consumer agents
        """
        # Cost function parameters
        self.a = 0.2
        self.b = 2
        self.c = 0

        # Environment bounds - NOW REPRESENT COLLECTIVE SYSTEM LIMITS
        self.max_gen_power = max_gen_power  # Total system generation limit
        self.min_gen_power = min_gen_power  # Total system generation minimum
        self.max_con_price = max_con_price  # Total system price limit
        self.min_con_price = min_con_price  # Total system price minimum
        self.max_steps = max_steps
        
        # Number of agents
        self.num_generators = num_generators
        self.num_consumers = num_consumers

        # Individual agent bounds (for action/observation spaces)
        # Each individual agent can range from 0 to the full system limit
        # but collectively they're constrained by the system bounds
        self.individual_max_gen_power = ind_max_gen_power  # Individual can't exceed system max
        self.individual_min_gen_power = ind_min_gen_power  # Individual can go to 0
        self.individual_max_con_price = ind_max_con_price  # Individual can't exceed system max
        self.individual_min_con_price = ind_min_con_price # Individual can go to 0

        # Variable to plot 
        self.cum_gen_power = []
        self.cum_con_price = []

        # Save printed states
        self.csv_initialized = False
        self.csv_file = f"market_log_all_episodes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

        self.iter_count = 0

        # Create agent names
        self.possible_agents = []
        for i in range(num_generators):
            self.possible_agents.append(f"generator_{i}")
        for i in range(num_consumers):
            self.possible_agents.append(f"consumer_{i}")
            
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        
        # Action spaces: 3 discrete actions (decrease, stay, increase)
        self.action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        
        # Observation space: Dictionary with own state + all other agents' states
        # Each agent observes: {"self": [value, profit], "agent_1": [value, profit], ...}
        max_agents = num_generators + num_consumers
        self.observation_spaces = {
            agent: Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(max_agents * 2,),  # Each agent has [value, profit]
                dtype=np.float32
            ) 
            for agent in self.possible_agents
        }

        # Action mappings
        self.action_step = 0.1
        self.action_to_delta = {
            0: -self.action_step,  # Decrease
            1: 0.0,                # Stay
            2: self.action_step    # Increase
        }
        
        # Initialize agent states
        self._generator_powers = {}
        self._consumer_prices = {}
        self.gen_profits = {}
        self.con_profits = {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.agents = self.possible_agents[:]
        
        # Initialize random positions within INDIVIDUAL bounds for all agents
        # The collective constraint will be enforced during stepping
        for i in range(self.num_generators):
            agent_name = f"generator_{i}"
            # Start with smaller individual values to avoid immediate constraint violations
            max_individual_start = self.max_gen_power / self.num_generators
            self._generator_powers[agent_name] = np.random.uniform(0, max_individual_start)
            
        for i in range(self.num_consumers):
            agent_name = f"consumer_{i}"
            # Start with smaller individual values to avoid immediate constraint violations
            max_individual_start = self.max_con_price / self.num_consumers
            self._consumer_prices[agent_name] = np.random.uniform(0, max_individual_start)
        
        # Ensure collective constraints are satisfied from the start
        self._enforce_collective_constraints()
        
        # Initialize step counter
        self.step_count = 0
        self.iter_count += 1
        
        # Calculate initial profits
        self._update_profits()
        
        # Get initial observations
        observations = self.observe()

        # Variable to plot 
        self.cum_gen_power = []
        self.cum_con_price = []
        
        return observations

    def _enforce_collective_constraints(self):
        """Enforce collective system constraints on all agents"""
        # Enforce generator collective constraints
        if self._generator_powers:
            total_gen_power = sum(self._generator_powers.values())
            
            # If total exceeds maximum, scale down proportionally
            if total_gen_power > self.max_gen_power:
                scale_factor = self.max_gen_power / total_gen_power
                for gen_name in self._generator_powers:
                    self._generator_powers[gen_name] *= scale_factor
            
            # If total is below minimum, scale up proportionally (if possible)
            elif total_gen_power < self.min_gen_power:
                scale_factor = self.min_gen_power / total_gen_power
                # Check if scaling would violate individual bounds
                max_individual_after_scaling = max(self._generator_powers.values()) * scale_factor
                if max_individual_after_scaling <= self.individual_max_gen_power:
                    for gen_name in self._generator_powers:
                        self._generator_powers[gen_name] *= scale_factor
                else:
                    # Distribute the minimum requirement more carefully
                    self._distribute_minimum_generation()
        
        # Enforce consumer collective constraints
        if self._consumer_prices:
            total_con_price = sum(self._consumer_prices.values())
            
            # If total exceeds maximum, scale down proportionally
            if total_con_price > self.max_con_price:
                scale_factor = self.max_con_price / total_con_price
                for con_name in self._consumer_prices:
                    self._consumer_prices[con_name] *= scale_factor
            
            # If total is below minimum, scale up proportionally (if possible)
            elif total_con_price < self.min_con_price and total_con_price > 0:
                scale_factor = self.min_con_price / total_con_price
                # Check if scaling would violate individual bounds
                max_individual_after_scaling = max(self._consumer_prices.values()) * scale_factor
                if max_individual_after_scaling <= self.individual_max_con_price:
                    for con_name in self._consumer_prices:
                        self._consumer_prices[con_name] *= scale_factor
                else:
                    # Distribute the minimum requirement more carefully
                    self._distribute_minimum_consumption()

    def _distribute_minimum_generation(self):
        """Distribute minimum generation requirement across generators"""
        if not self._generator_powers:
            return
            
        # Set all to minimum viable distribution
        min_per_generator = self.min_gen_power / self.num_generators
        for gen_name in self._generator_powers:
            self._generator_powers[gen_name] = min(min_per_generator, self.individual_max_gen_power)

    def _distribute_minimum_consumption(self):
        """Distribute minimum consumption requirement across consumers"""
        if not self._consumer_prices:
            return
            
        # Set all to minimum viable distribution
        min_per_consumer = self.min_con_price / self.num_consumers
        for con_name in self._consumer_prices:
            self._consumer_prices[con_name] = min(min_per_consumer, self.individual_max_con_price)

    def observe(self):
        """Generate observations for all agents"""
        observations = {}
        
        # Create ordered list of all agents for consistent observation structure
        all_agents = self.possible_agents[:]
        
        for observer_agent in self.agents:
            obs_vector = []
            
            # First, add observer's own state
            if observer_agent.startswith("generator"):
                own_value = self._generator_powers.get(observer_agent, 0)
                own_profit = self.gen_profits.get(observer_agent, 0)
            else:  # consumer
                own_value = self._consumer_prices.get(observer_agent, 0)
                own_profit = self.con_profits.get(observer_agent, 0)
            
            obs_vector.extend([own_value, own_profit])
            
            # Then add all other agents' states in consistent order
            for other_agent in all_agents:
                if other_agent == observer_agent:
                    continue  # Skip self, already added
                    
                if other_agent.startswith("generator"):
                    other_value = self._generator_powers.get(other_agent, 0)
                    other_profit = self.gen_profits.get(other_agent, 0)
                else:  # consumer
                    other_value = self._consumer_prices.get(other_agent, 0)
                    other_profit = self.con_profits.get(other_agent, 0)
                
                obs_vector.extend([other_value, other_profit])
            
            observations[observer_agent] = np.array(obs_vector, dtype=np.float32)
        
        return observations

    def step(self, actions):
        """Execute one step of the environment"""
        # Handle empty actions
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # Format actions if needed
        formatted_actions = self._format_actions(actions)
        
        # Apply actions simultaneously with individual bounds first
        for agent_name, action in formatted_actions.items():
            if agent_name not in self.agents:
                continue
                
            delta = self.action_to_delta[action]
            
            if agent_name.startswith("generator"):
                # Apply individual bounds first
                new_value = self._generator_powers[agent_name] + delta
                self._generator_powers[agent_name] = np.clip(
                    new_value, 
                    self.individual_min_gen_power, 
                    self.individual_max_gen_power
                )
            elif agent_name.startswith("consumer"):
                # Apply individual bounds first
                new_value = self._consumer_prices[agent_name] + delta
                self._consumer_prices[agent_name] = np.clip(
                    new_value, 
                    self.individual_min_con_price, 
                    self.individual_max_con_price
                )
        
        # CRUCIAL: Enforce collective constraints after individual actions
        self._enforce_collective_constraints()
        
        # Update step counter
        self.step_count += 1
        
        # Calculate new profits
        self._update_profits()
        
        # Calculate rewards and check for termination
        rewards = self._calculate_rewards()
        dones = self._check_termination()
        
        # Get new observations
        observations = self.observe()
        
        # Info dictionary - include collective constraint info
        infos = {}
        for agent in self.agents:
            infos[agent] = {
                "generator_powers": dict(self._generator_powers),
                "consumer_prices": dict(self._consumer_prices),
                "generator_profits": dict(self.gen_profits),
                "consumer_profits": dict(self.con_profits),
                "total_generation": sum(self._generator_powers.values()),
                "total_consumption_price": sum(self._consumer_prices.values()),
                "generation_constraint_active": sum(self._generator_powers.values()) >= self.max_gen_power * 0.99,
                "consumption_constraint_active": sum(self._consumer_prices.values()) >= self.max_con_price * 0.99,
                "step": self.step_count
            }
        
        # Remove agents if game is done
        if any(dones.values()):
            self.agents = []
        
        return observations, rewards, dones, infos

    def _update_profits(self):
        """Calculate profits for all agents"""
        # Clear previous profits
        self.gen_profits = {}
        self.con_profits = {}
        
        # Calculate total power and average price for market clearing
        total_power = sum(self._generator_powers.values())
        avg_price = np.mean(list(self._consumer_prices.values())) if self._consumer_prices else 0
        
        # Calculate generator profits
        for gen_name, power in self._generator_powers.items():
            # Generator cost function: Hg = a * P^2 + b * P + c
            Hg = self.a * power**2 + self.b * power + self.c
            # Generator profit: P * avg_price - cost
            self.gen_profits[gen_name] = power * avg_price - Hg
        
        # Calculate consumer profits
        for con_name, price in self._consumer_prices.items():
            # Consumer utility: share of total power * (1/ln(1+price))
            # Adding small epsilon to avoid division by zero
            epsilon = 1e-8
            power_share = total_power / max(self.num_consumers, 1)  # Equal sharing of total power
            self.con_profits[con_name] = power_share * (1 / np.log(1 + price + epsilon))

    def _calculate_rewards(self):
        """Calculate rewards for all agents"""
        rewards = {}
        
        # Assign rewards based on profits
        for gen_name, profit in self.gen_profits.items():
            rewards[gen_name] = profit
            
        for con_name, profit in self.con_profits.items():
            rewards[con_name] = profit
        
        return rewards

    def _check_termination(self):
        """Check if the episode should terminate"""
        # Terminate if any agent reaches profit threshold or max steps reached
        all_profits = list(self.gen_profits.values()) + list(self.con_profits.values())
        max_steps_reached = self.step_count >= self.max_steps
        
        terminated = max_steps_reached
        
        # Return termination status for all agents
        return {agent: terminated for agent in self.agents}

    def _format_actions(self, actions):
        """Convert different action formats to expected dict format"""
        # If actions is already a dict with agent keys, return as is
        if isinstance(actions, dict) and all(agent in actions for agent in self.agents):
            return actions
        
        # If actions is a list, map to agents by index
        if isinstance(actions, (list, tuple)):
            formatted = {}
            for i, agent in enumerate(self.agents):
                if i < len(actions):
                    formatted[agent] = actions[i]
            return formatted
        
        # If actions is a dict with policy IDs
        if isinstance(actions, dict):
            formatted = {}
            policy_keys = list(actions.keys())
            
            if len(policy_keys) == len(self.agents):
                for i, agent in enumerate(self.agents):
                    if i < len(policy_keys):
                        formatted[agent] = actions[policy_keys[i]]
            return formatted
        
        # Single agent case
        if len(self.agents) == 1:
            return {self.agents[0]: actions}
        
        print(f"Warning: Could not format actions {actions} of type {type(actions)}")
        return {}
    
    def _init_csv_log(self):
        if not self.csv_initialized:
            fieldnames = ['episode', 'step', 'total_generation', 'total_consumption_price']
            
            # Add fields for each generator
            for i in range(self.num_generators):
                fieldnames.extend([f'generator_{i}_power', f'generator_{i}_profit'])
            
            # Add fields for each consumer
            for i in range(self.num_consumers):
                fieldnames.extend([f'consumer_{i}_price', f'consumer_{i}_profit'])
            
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
            self.csv_initialized = True

    def render(self, mode="human"):
        """Render the environment state"""
        if mode == "human":
            self.print_state()

    def print_state(self):
        """Print current state of the energy market and log to CSV"""
        self._init_csv_log()
        
        # Calculate totals
        total_generation = sum(self._generator_powers.values())
        total_consumption_price = sum(self._consumer_prices.values())
        
        # Prepare state info for CSV
        state_info = {
            'episode': self.iter_count,
            'step': self.step_count,
            'total_generation': total_generation,
            'total_consumption_price': total_consumption_price
        }
        
        # Add generator data
        for i in range(self.num_generators):
            agent_name = f'generator_{i}'
            state_info[f'generator_{i}_power'] = self._generator_powers.get(agent_name, 0)
            state_info[f'generator_{i}_profit'] = self.gen_profits.get(agent_name, 0)
        
        # Add consumer data
        for i in range(self.num_consumers):
            agent_name = f'consumer_{i}'
            state_info[f'consumer_{i}_price'] = self._consumer_prices.get(agent_name, 0)
            state_info[f'consumer_{i}_profit'] = self.con_profits.get(agent_name, 0)

        # Print summary
        # print(f"Step {self.step_count}: Total Gen={total_generation:.3f}/{self.max_gen_power}, Total Price={total_consumption_price:.3f}/{self.max_con_price}")

        # Write to CSV
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=state_info.keys())
            writer.writerow(state_info)

    def close(self):
        """Clean up environment resources"""
        pass