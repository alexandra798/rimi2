"""Token系统和RPN验证器"""
from enum import Enum
import numpy as np


class TokenType(Enum):
    SPECIAL = "special"  # BEG, END
    OPERAND = "operand"  # 操作数
    OPERATOR = "operator"  # 操作符


class Token:
    def __init__(self, token_type, name, value=None, arity=0):
        self.type = token_type
        self.name = name
        self.value = value
        self.arity = arity  # 操作符需要的操作数个数


# Token定义字典
TOKEN_DEFINITIONS = {
    # 特殊标记
    'BEG': Token(TokenType.SPECIAL, 'BEG'),
    'END': Token(TokenType.SPECIAL, 'END'),

    # 操作数 - 股票特征
    'open': Token(TokenType.OPERAND, 'open'),
    'high': Token(TokenType.OPERAND, 'high'),
    'low': Token(TokenType.OPERAND, 'low'),
    'close': Token(TokenType.OPERAND, 'close'),
    'volume': Token(TokenType.OPERAND, 'volume'),
    'vwap': Token(TokenType.OPERAND, 'vwap'),

    # 操作数 - 时间窗口（7个，用于时序操作）
    'delta_1': Token(TokenType.OPERAND, 'delta_1', value=1),
    'delta_5': Token(TokenType.OPERAND, 'delta_5', value=5),
    'delta_10': Token(TokenType.OPERAND, 'delta_10', value=10),
    'delta_20': Token(TokenType.OPERAND, 'delta_20', value=20),
    'delta_30': Token(TokenType.OPERAND, 'delta_30', value=30),
    'delta_40': Token(TokenType.OPERAND, 'delta_40', value=40),
    'delta_50': Token(TokenType.OPERAND, 'delta_50', value=50),

    # 操作数 - 常数（13个，根据论文Table 3）
    'const_-30': Token(TokenType.OPERAND, 'const_-30', value=-30.0),
    'const_-10': Token(TokenType.OPERAND, 'const_-10', value=-10.0),
    'const_-5': Token(TokenType.OPERAND, 'const_-5', value=-5.0),
    'const_-2': Token(TokenType.OPERAND, 'const_-2', value=-2.0),
    'const_-1': Token(TokenType.OPERAND, 'const_-1', value=-1.0),
    'const_-0.5': Token(TokenType.OPERAND, 'const_-0.5', value=-0.5),
    'const_-0.01': Token(TokenType.OPERAND, 'const_-0.01', value=-0.01),
    'const_0.5': Token(TokenType.OPERAND, 'const_0.5', value=0.5),
    'const_1': Token(TokenType.OPERAND, 'const_1', value=1.0),
    'const_2': Token(TokenType.OPERAND, 'const_2', value=2.0),
    'const_5': Token(TokenType.OPERAND, 'const_5', value=5.0),
    'const_10': Token(TokenType.OPERAND, 'const_10', value=10.0),
    'const_30': Token(TokenType.OPERAND, 'const_30', value=30.0),

    # 一元操作符 - 横截面（4个）
    'sign': Token(TokenType.OPERATOR, 'sign', arity=1),
    'abs': Token(TokenType.OPERATOR, 'abs', arity=1),
    'log': Token(TokenType.OPERATOR, 'log', arity=1),
    'csrank': Token(TokenType.OPERATOR, 'csrank', arity=1),  # 横截面排名

    # 二元操作符（需要2个操作数）
    'add': Token(TokenType.OPERATOR, 'add', arity=2),  # +
    'sub': Token(TokenType.OPERATOR, 'sub', arity=2),  # -
    'mul': Token(TokenType.OPERATOR, 'mul', arity=2),  # *
    'div': Token(TokenType.OPERATOR, 'div', arity=2),  # /
    'greater': Token(TokenType.OPERATOR, 'greater', arity=2),
    'less': Token(TokenType.OPERATOR, 'less', arity=2),

    # 时序操作符 - 特殊处理（实际使用时需要delta参数）
    'ts_ref': Token(TokenType.OPERATOR, 'ts_ref', arity=1),  # Ref(x,t)
    'ts_rank': Token(TokenType.OPERATOR, 'ts_rank', arity=1),  # Rank(x,t)
    'ts_mean': Token(TokenType.OPERATOR, 'ts_mean', arity=1),  # Mean(x,t)
    'ts_med': Token(TokenType.OPERATOR, 'ts_med', arity=1),  # Med(x,t)
    'ts_sum': Token(TokenType.OPERATOR, 'ts_sum', arity=1),  # Sum(x,t)
    'ts_std': Token(TokenType.OPERATOR, 'ts_std', arity=1),  # Std(x,t)
    'ts_var': Token(TokenType.OPERATOR, 'ts_var', arity=1),  # Var(x,t)
    'ts_max': Token(TokenType.OPERATOR, 'ts_max', arity=1),  # Max(x,t)
    'ts_min': Token(TokenType.OPERATOR, 'ts_min', arity=1),  # Min(x,t)
    'ts_skew': Token(TokenType.OPERATOR, 'ts_skew', arity=1),  # Skew(x,t)
    'ts_kurt': Token(TokenType.OPERATOR, 'ts_kurt', arity=1),  # Kurt(x,t)
    'ts_wma': Token(TokenType.OPERATOR, 'ts_wma', arity=1),  # WMA(x,t)
    'ts_ema': Token(TokenType.OPERATOR, 'ts_ema', arity=1),  # EMA(x,t)


    # 相关性操作符（需要3个操作数：2个数据，1个时间窗口）
    'corr': Token(TokenType.OPERATOR, 'corr', arity=3),
    'cov': Token(TokenType.OPERATOR, 'cov', arity=3),
}

# 创建Token索引映射
TOKEN_TO_INDEX = {name: idx for idx, name in enumerate(TOKEN_DEFINITIONS.keys())}
INDEX_TO_TOKEN = {idx: name for name, idx in TOKEN_TO_INDEX.items()}
TOTAL_TOKENS = len(TOKEN_DEFINITIONS)


class RPNValidator:
    """验证和评估逆波兰表达式"""

    @staticmethod
    def is_valid_partial_expression(token_sequence):
        """检查是否为合法的部分RPN表达式"""
        if not token_sequence or token_sequence[0].name != 'BEG':
            return False

        stack_size = 0
        i = 1  # 跳过BEG

        while i < len(token_sequence):
            token = token_sequence[i]

            if token.name == 'END':
                return stack_size == 1

            if token.type == TokenType.OPERAND:
                # delta不入栈（作为参数）
                if not token.name.startswith('delta_'):
                    stack_size += 1
            elif token.type == TokenType.OPERATOR:
                # 检查是否需要delta参数
                if token.name in ['ts_ref', 'ts_rank'] or token.name.startswith('ts_'):
                    # 这些操作符需要delta参数
                    if i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                        i += 1  # 跳过delta
                    # 栈操作：消耗arity个，产生1个
                    if stack_size < token.arity:
                        return False
                    stack_size = stack_size - token.arity + 1
                else:
                    # 普通操作符
                    if stack_size < token.arity:
                        return False
                    stack_size = stack_size - token.arity + 1

            i += 1

        return stack_size >= 1

    @staticmethod
    def get_valid_next_tokens(token_sequence):
        """返回当前状态下所有合法的下一个Token"""
        if not token_sequence:
            return ['BEG']

        if len(token_sequence) >= 30:
            return ['END'] if RPNValidator.can_terminate(token_sequence) else []

        last_token = token_sequence[-1] if token_sequence else None

        # 如果最后一个是需要时间参数的操作符，下一个必须是delta
        time_ops = ['ts_ref', 'ts_rank', 'ts_mean', 'ts_med', 'ts_sum', 'ts_std',
                    'ts_var', 'ts_max', 'ts_min', 'ts_skew', 'ts_kurt',
                    'ts_wma', 'ts_ema']
        if last_token and last_token.name in time_ops:
            return ['delta_1', 'delta_5', 'delta_10', 'delta_20',
                    'delta_30', 'delta_40', 'delta_50']

        # 计算当前栈大小
        stack_size = RPNValidator.calculate_stack_size(token_sequence)
        valid_tokens = []

        # 操作数总是可以添加（除非栈溢出）
        if stack_size < 10:
            # 添加特征
            valid_tokens.extend(['open', 'high', 'low', 'close', 'volume', 'vwap'])
            # 添加常数
            valid_tokens.extend(['const_-30', 'const_-10', 'const_-5', 'const_-2',
                                 'const_-1', 'const_-0.5', 'const_-0.01', 'const_0.5',
                                 'const_1', 'const_2', 'const_5', 'const_10', 'const_30'])

        # 操作符需要足够的操作数
        for token_name, token in TOKEN_DEFINITIONS.items():
            if token.type == TokenType.OPERATOR:
                if token.arity <= stack_size:
                    valid_tokens.append(token_name)

        # END需要栈中恰好1个元素
        if stack_size == 1 and len(token_sequence) > 3:
            valid_tokens.append('END')

        return valid_tokens

    @staticmethod
    def calculate_stack_size(token_sequence):
        """计算当前栈中的元素数量"""
        stack_size = 0
        i = 1  # 跳过BEG

        while i < len(token_sequence):
            token = token_sequence[i]

            if token.name == 'END':
                break

            if token.type == TokenType.OPERAND:
                # delta不入栈
                if not token.name.startswith('delta_'):
                    stack_size += 1
            elif token.type == TokenType.OPERATOR:
                # 检查是否有delta参数
                time_ops = ['ts_ref', 'ts_rank'] + [f'ts_{op}' for op in
                                              ['mean', 'med', 'sum', 'std', 'var', 'max', 'min',
                                               'skew', 'kurt', 'wma', 'ema']]
                if token.name in time_ops:
                    if i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                        i += 1  # 跳过delta

                # 更新栈大小
                stack_size = stack_size - token.arity + 1

            i += 1

        return stack_size

    @staticmethod
    def can_terminate(token_sequence):
        """检查是否可以终止（栈中正好剩1个操作数）"""
        return RPNValidator.calculate_stack_size(token_sequence) == 1


class FormulaGenerator:
    """基于Token的公式生成器"""

    def __init__(self):
        self.token_sequence = [TOKEN_DEFINITIONS['BEG']]
        self.operand_stack_count = 0

    def add_next_token(self, token_name):
        """逐个添加Token构建公式"""
        if token_name not in TOKEN_DEFINITIONS:
            raise ValueError(f"Unknown token: {token_name}")

        token = TOKEN_DEFINITIONS[token_name]
        self.token_sequence.append(token)
        self.update_stack_count(token)
        return token

    def update_stack_count(self, token):
        """更新栈计数"""
        if token.type == TokenType.OPERAND:
            self.operand_stack_count += 1
        elif token.type == TokenType.OPERATOR:
            self.operand_stack_count = self.operand_stack_count - token.arity + 1

    def can_terminate(self):
        """检查是否可以结束（栈中正好剩1个操作数）"""
        return self.operand_stack_count == 1

    def get_valid_actions(self):
        """获取当前状态下的合法动作"""
        return RPNValidator.get_valid_next_tokens(self.token_sequence)

    def to_formula_string(self):
        """将Token序列转换为可读的公式字符串"""
        # 简化版本：直接返回token名称序列
        return ' '.join([t.name for t in self.token_sequence[1:] if t.name != 'END'])