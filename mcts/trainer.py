"""RiskMiner完整训练器"""
import logging
import numpy as np
import sys
import os
import torch

from alpha.evaluator import FormulaEvaluator
from core import TOKEN_DEFINITIONS
from mcts.reward_calculator import RewardCalculator

from mcts.node import MCTSNode
from mcts.searcher import MCTSSearcher
from mcts.environment import AlphaMiningMDP, MDPState


from policy.network import PolicyNetwork
from policy.optimizer import RiskSeekingOptimizer

logger = logging.getLogger(__name__)


class RiskMinerTrainer:

    def __init__(self, X_data, y_data, device=None, use_sampling=True, sample_size=50000, random_seed=42):
        self.X_data = X_data
        self.y_data = y_data
        self.use_sampling = use_sampling

        # 如果数据太大，创建采样版本用于训练
        if use_sampling and len(X_data) > sample_size:
            logger.info(f"Data too large ({len(X_data)} rows), creating sampled version...")
            np.random.seed(random_seed)
            sample_indices = np.random.choice(len(X_data), sample_size, replace=False)
            self.X_train_sample = X_data.iloc[sample_indices]
            self.y_train_sample = y_data.iloc[sample_indices]
            logger.info(f"Using {sample_size} samples for MCTS training (seed={random_seed})")
        else:
            self.X_train_sample = X_data
            self.y_train_sample = y_data

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 初始化组件
        self.mdp_env = AlphaMiningMDP()
        self.policy_network = PolicyNetwork().to(self.device)
        self.optimizer = RiskSeekingOptimizer(self.policy_network, device=self.device)
        self.mcts_searcher = MCTSSearcher(self.policy_network, device=self.device)
        self.alpha_pool = []
        self.reward_calculator = RewardCalculator(self.alpha_pool, random_seed=random_seed)
        self.formula_evaluator = FormulaEvaluator()  # 使用统一的评估器

        logger.info(f"Policy network moved to {self.device}")


    def train(self, num_iterations=200, num_simulations_per_iteration=50):
        """主训练循环"""

        for iteration in range(num_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

            # 每10轮清理一次缓存
            if iteration > 0 and iteration % 10 == 0:
                # 清理FormulaEvaluator缓存
                self.formula_evaluator.clear_cache()
                # 清理RewardCalculator缓存
                self.reward_calculator._cache.clear()
                logger.info("Cleared caches to free memory")

            # 阶段1：MCTS搜索收集轨迹
            trajectories = self.collect_trajectories_with_mcts(
                num_episodes=10,
                num_simulations_per_episode=num_simulations_per_iteration
            )

            # 阶段2：使用收集的轨迹训练策略网络（如果启用）
            if self.optimizer and trajectories:
                avg_loss = self.train_policy_network(trajectories)
                logger.info(f"Policy network loss: {avg_loss:.4f}")

            # 阶段3：评估和更新Alpha池
            self.update_alpha_pool(trajectories, iteration)

            # 打印统计信息
            if (iteration + 1) % 10 == 0:
                self.print_statistics()

    def search_one_iteration(self, root):
        """执行一次MCTS搜索迭代"""
        return self.mcts_searcher.search_one_iteration(
            root_node=root,
            mdp_env=self.mdp_env,
            reward_calculator=self.reward_calculator,
            X_data=self.X_train_sample,  # 使用采样集而非全量
            y_data=self.y_train_sample  # 使用采样集而非全量
        )

    def collect_trajectories_with_mcts(self, num_episodes, num_simulations_per_episode):
        """使用MCTS收集训练轨迹"""
        all_trajectories = []

        for episode in range(num_episodes):
            logger.debug(f"Starting episode {episode + 1}/{num_episodes}")
            # 重置环境
            initial_state = self.mdp_env.reset()

            # 创建根节点
            root = MCTSNode(state=initial_state)

            # 执行MCTS搜索
            for sim in range(num_simulations_per_episode):
                if sim % 10 == 0:
                    logger.debug(f"  Simulation {sim}/{num_simulations_per_episode}")
                trajectory = self.search_one_iteration(root)

            # 选择最佳动作序列作为最终轨迹
            final_trajectory = self.extract_best_trajectory(root)
            if final_trajectory:
                all_trajectories.append(final_trajectory)
                # 记录生成的公式
                formula_str = self.get_formula_from_trajectory(final_trajectory)
                if formula_str:
                    logger.debug(f"Episode {episode + 1}: {formula_str}")

        return all_trajectories



    def train_policy_network(self, trajectories):
        """训练策略网络"""
        if not self.optimizer:
            return 0.0

        total_loss = 0
        num_updates = 0

        for trajectory in trajectories:
            if trajectory:  # 确保轨迹非空
                loss = self.optimizer.train_on_episode(trajectory)
                if loss > 0:  # 只有超过分位数的轨迹才会产生损失
                    total_loss += loss
                    num_updates += 1

        avg_loss = total_loss / max(num_updates, 1)

        if num_updates > 0:
            logger.info(f"Policy network updated: {num_updates}/{len(trajectories)} episodes")
            logger.info(f"Current quantile: {self.optimizer.quantile_estimate:.4f}")

        return avg_loss

    def extract_best_trajectory(self, root):
        """从MCTS树中提取最佳轨迹"""
        trajectory = []
        current = root
        max_depth = 30
        depth = 0
        from core import RPNValidator

        while current.children and not current.is_terminal() and depth < max_depth:
            # 选择访问次数最多的子节点
            best_action = None
            best_visits = -1
            for action, child in current.children.items():
                if child.N > best_visits:
                    best_visits = child.N
                    best_action = action
                    best_child = child

            if best_action is None:
                break

            # 计算这一步的奖励
            if best_child.state.token_sequence[-1].name == 'END':
                reward = self.reward_calculator.calculate_terminal_reward(
                    best_child.state, self.X_train_sample, self.y_train_sample
                )

            else:
                reward = self.reward_calculator.calculate_intermediate_reward(
                    best_child.state, self.X_train_sample, self.y_train_sample
                )

            valid_now = RPNValidator.get_valid_next_tokens(current.state.token_sequence)
            if best_action not in valid_now:
                # 非法动作，停止添加
                logger.debug(f"Illegal action {best_action} detected, stopping trajectory")
                break

            trajectory.append((current.state, best_action, reward))
            current = best_child
            depth += 1

        if not current.is_terminal() and depth < max_depth:
            valid_actions = RPNValidator.get_valid_next_tokens(current.state.token_sequence)

            # 检查是否需要补充delta参数
            if valid_actions and all(a.startswith('delta_') for a in valid_actions):
                # 选择满足最小窗口要求的delta
                last_token = current.state.token_sequence[-1]
                min_window = TOKEN_DEFINITIONS[last_token.name].min_window or 3

                suitable_delta = None
                for delta in valid_actions:
                    delta_value = TOKEN_DEFINITIONS[delta].value
                    if delta_value >= min_window:
                        suitable_delta = delta
                        break

                if suitable_delta:
                    # 添加delta
                    delta_state = current.state.copy()
                    delta_state.add_token(suitable_delta)
                    delta_reward = self.reward_calculator.calculate_intermediate_reward(
                        delta_state, self.X_train_sample, self.y_train_sample
                    )
                    trajectory.append((current.state, suitable_delta, delta_reward))
                    # 不要改变current的类型，使用delta_state作为新的状态

                    # 现在尝试添加END，使用delta_state而不是current
                    if RPNValidator.can_terminate(delta_state.token_sequence):
                        terminal_state = delta_state.copy()
                        terminal_state.add_token('END')
                        terminal_reward = self.reward_calculator.calculate_terminal_reward(
                            terminal_state, self.X_train_sample, self.y_train_sample
                        )
                        trajectory.append((delta_state, 'END', terminal_reward))
                else:
                    # 如果没有delta，直接检查current.state
                    if RPNValidator.can_terminate(current.state.token_sequence):
                        terminal_state = current.state.copy()
                        terminal_state.add_token('END')
                        terminal_reward = self.reward_calculator.calculate_terminal_reward(
                            terminal_state, self.X_train_sample, self.y_train_sample
                        )
                        trajectory.append((current.state, 'END', terminal_reward))

        return trajectory

    def get_formula_from_trajectory(self, trajectory):
        """从轨迹中获取公式字符串"""
        if not trajectory:
            return None

        # 构建完整状态
        state = MDPState()
        for s, action, r in trajectory:
            state.add_token(action)

        # 转换为可读公式
        return state.token_sequence

    def update_alpha_pool(self, trajectories, iteration):
        """更新Alpha池 - 增强版，过滤常数"""
        new_formulas = []
        constant_count = 0

        for trajectory in trajectories:
            if not trajectory:
                continue

            # 构建完整状态
            final_state = MDPState()
            for state, action, reward in trajectory:
                final_state.add_token(action)

            # 检查是否以END结束
            if final_state.token_sequence[-1].name == 'END':
                formula_rpn = ' '.join([t.name for t in final_state.token_sequence])
                alpha_values = self.formula_evaluator.evaluate(formula_rpn, self.X_train_sample)

                if alpha_values is not None and not alpha_values.isna().all():
                    # 新增：检查是否为常数
                    if hasattr(alpha_values, 'values'):
                        values = alpha_values.values
                    else:
                        values = np.array(alpha_values)

                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 10:  # 足够的有效值
                        std = np.std(valid_values)

                        if std < 1e-6:  # 常数检测
                            constant_count += 1
                            logger.debug(f"Skipping constant formula: {formula_rpn[:50]}...")
                            continue

                        # 计算IC
                        ic = self.reward_calculator.calculate_ic(alpha_values, self.y_data)

                        # 只添加IC足够高的公式
                        if abs(ic) >= 0.01:
                            new_formulas.append({
                                'formula': formula_rpn,
                                'ic': ic,
                                'values': alpha_values,
                                'iteration': iteration
                            })

        # 添加新公式到池中
        for formula_info in new_formulas:
            exists = any(a['formula'] == formula_info['formula'] for a in self.alpha_pool)
            if not exists:
                self.alpha_pool.append(formula_info)
                logger.info(f"New valid formula: {formula_info['formula'][:50]}... IC={formula_info['ic']:.4f}")

        # 保持池大小
        if len(self.alpha_pool) > 100:
            # 移除IC最低的
            self.alpha_pool.sort(key=lambda x: abs(x['ic']), reverse=True)
            self.alpha_pool = self.alpha_pool[:100]

        if constant_count > 0:
            logger.info(f"Filtered out {constant_count} constant formulas in iteration {iteration}")

    def print_statistics(self):
        """打印训练统计信息"""
        logger.info("\n=== Training Statistics ===")
        logger.info(f"Alpha pool size: {len(self.alpha_pool)}")

        if self.alpha_pool:
            top_5 = sorted(self.alpha_pool, key=lambda x: x['ic'], reverse=True)[:5]
            logger.info("\nTop 5 Alphas:")
            for i, alpha in enumerate(top_5, 1):
                formula = alpha['formula']
                if len(formula) > 80:
                    formula = formula[:77] + "..."
                logger.info(f"{i}. Formula: {formula}")
                logger.info(f"   IC: {alpha['ic']:.4f}")

        if self.optimizer:
            logger.info(f"\nCurrent quantile estimate: {self.optimizer.quantile_estimate:.4f}")

    def get_top_formulas(self, n=5):
        """获取最佳的n个公式"""
        if not self.alpha_pool:
            return []

        sorted_pool = sorted(self.alpha_pool, key=lambda x: x['ic'], reverse=True)
        return [alpha['formula'] for alpha in sorted_pool[:n]]