import random
import json
import os
import numpy as np
import plotly.graph_objects as go
from numba import njit, prange

# 预计算概率表
RATE_5_STAR = np.zeros(81)
for i in range(81):
    if i < 66: RATE_5_STAR[i] = 0.008
    elif i < 71: RATE_5_STAR[i] = 0.008 + 0.04 * (i - 65)
    elif i < 76: RATE_5_STAR[i] = 0.208 + 0.08 * (i - 70)
    elif i < 79: RATE_5_STAR[i] = 0.608 + 0.1 * (i - 75)
    else: RATE_5_STAR[i] = 1.0

RATE_4_STAR = np.zeros(12)
for i in range(12):
    if i < 10: RATE_4_STAR[i] = 0.06
    else: RATE_4_STAR[i] = 1.0

# 抽取结果常量
OUTCOME_3_STAR = 30
OUTCOME_4_STAR_UP = 42
OUTCOME_4_STAR_OTHER = 41
OUTCOME_4_STAR_WEAPON = 43
OUTCOME_5_STAR_UP = 52
OUTCOME_5_STAR_STANDARD = 51

@njit
def core_pull(pity_5, pity_4, is_guaranteed, is_4star_guaranteed, banner_type):
    """
    核心抽卡逻辑 (Numba加速)
    banner_type: 0-角色池, 1-武器池
    返回: (outcome_type, new_pity_5, new_pity_4, new_is_guaranteed, new_is_4star_guaranteed)
    """
    r = random.random()
    
    # 检查五星
    current_pity_5 = pity_5 + 1
    rate_5 = 1.0
    if current_pity_5 <= 80:
        rate_5 = RATE_5_STAR[current_pity_5]
        
    if rate_5 > r:
        # 抽到五星
        new_pity_5 = 0
        # 特殊规则：若同时满足四星保底(pity_4==9)，重置四星保底
        new_pity_4 = 0 if pity_4 == 9 else pity_4
        
        if banner_type == 1:
            # 武器池：必定UP
            return OUTCOME_5_STAR_UP, new_pity_5, new_pity_4, is_guaranteed, is_4star_guaranteed
        else:
            # 角色池
            if not is_guaranteed and random.random() < 0.5:
                # 歪常驻
                return OUTCOME_5_STAR_STANDARD, new_pity_5, new_pity_4, True, is_4star_guaranteed
            else:
                # 中UP
                return OUTCOME_5_STAR_UP, new_pity_5, new_pity_4, False, is_4star_guaranteed

    # 检查四星
    current_pity_4 = pity_4 + 1
    rate_4 = 1.0
    if current_pity_4 <= 11:
        rate_4 = RATE_4_STAR[current_pity_4]
        
    if rate_4 > r:
        # 抽到四星
        new_pity_4 = 0
        new_pity_5 = pity_5 + 1
        
        if banner_type == 0:
            # 角色池：应用4星保底逻辑
            if is_4star_guaranteed:
                return OUTCOME_4_STAR_UP, new_pity_5, new_pity_4, is_guaranteed, False
            else:
                if random.random() < 0.5:
                    return OUTCOME_4_STAR_OTHER, new_pity_5, new_pity_4, is_guaranteed, True
                else:
                    return OUTCOME_4_STAR_UP, new_pity_5, new_pity_4, is_guaranteed, False
        else:
            # 武器池：应用4星保底逻辑
            if is_4star_guaranteed:
                return OUTCOME_4_STAR_UP, new_pity_5, new_pity_4, is_guaranteed, False
            else:
                if random.random() < 0.5:
                    return OUTCOME_4_STAR_OTHER, new_pity_5, new_pity_4, is_guaranteed, True
                else:
                    return OUTCOME_4_STAR_UP, new_pity_5, new_pity_4, is_guaranteed, False
            
    # 三星
    return OUTCOME_3_STAR, pity_5 + 1, pity_4 + 1, is_guaranteed, is_4star_guaranteed

class GachaSimulator:
    def __init__(self, initial_guaranteed=False, initial_coral=0, initial_pity_5star=0, initial_pity_weapon=0):
        # 定义所有实例属性并初始化
        self.initial_guaranteed = initial_guaranteed
        self.initial_coral = initial_coral
        self.initial_pity_5star = initial_pity_5star
        self.initial_pity_weapon = initial_pity_weapon

        # 五星限定角色是否保底
        self.featured_5star_guaranteed = False
        # 四星限定角色是否保底
        self.featured_4star_guaranteed = False
        # 四星限定武器是否保底
        self.featured_4star_weapon_guaranteed = False
        # 距上个五星的抽数
        self.pity_5star = 0
        # 常驻池五星角色及已有数、抽到数：'角色名': [链数, 抽到]
        self.standard_5stars = {}
        # 限定池五星角色及已有数、抽到数：'角色名': [链数, 抽到]
        self.featured_5stars = {}
        # 本期概率up的限定五星角色
        self.rate_up_5star = ''
        # 距上个四星的抽数
        self.pity_4star = 0
        # 四星角色及已有数、抽到数：'角色名': [链数, 抽到]
        self._4stars = {}
        # 本期概率up的四星角色
        self.rate_up_4stars = []
        # 抽取的三星武器数量
        self.weapon = 0
        # 本次抽取事件中获取的余波珊瑚（大珊瑚）
        self.total_afterglow_coral_count = 0
        # 本次抽取事件中获取的残振珊瑚（小珊瑚）
        self.total_oscillated_coral_count = 0
        # 总抽数
        self.pull_count = 0

        # 武器池相关属性
        self.pity_5star_weapon = 0
        self.pity_4star_weapon = 0
        self.featured_weapon_count = 0

        # 调用reset方法初始化
        self.reset()

    def reset(self):
        """
        重置所有状态到初始值
        """
        self.featured_5star_guaranteed = self.initial_guaranteed
        self.featured_4star_guaranteed = False
        self.featured_4star_weapon_guaranteed = False
        self.pity_5star = self.initial_pity_5star
        
        # 默认配置
        self.standard_5stars = {
            '凌阳': [-1, 0],
            '安可': [-1, 0],
            '卡卡罗': [-1, 0],
            '鉴心': [-1, 0],
            '维里奈': [-1, 0],
        }
        self.rate_up_5star = '当期限定五星'
        self.featured_5stars = {
            self.rate_up_5star: [-1, 0]
        }
        self._4stars = {
            '散华': [4, 0],
            '白芷': [-1, 0],
            '釉瑚': [-1, 0],
            '炽霞': [0, 0],
            '莫特斐': [-1, 0],
            '渊武': [-1, 0],
            '灯灯': [-1, 0],
            '泱泱': [-1, 0],
            '秋水': [-1, 0],
            '桃祈': [-1, 0],
            '丹瑾': [0, 0],
            '卜灵': [-1, 0]
        }
        self.rate_up_4stars = ['炽霞', '丹瑾', '卜灵']

        # 尝试加载配置文件
        config_path = 'gacha_config.json'
        config_data = None
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                print(f"成功加载配置文件: {config_path}")
            except Exception as e:
                print(f"加载配置文件失败: {e}，将使用默认配置。")
        else:
            print(f"配置文件不存在: {config_path}，将使用默认配置。")

        if config_data:
            # 从配置加载并覆盖默认值
            if 'rate_up_4stars' in config_data:
                self.rate_up_4stars = config_data['rate_up_4stars']
            
            # 如果配置中有 standard_5stars，则更新
            if 'standard_5stars' in config_data:
                # 兼容字典或列表格式
                if isinstance(config_data['standard_5stars'], dict):
                    for k, v in config_data['standard_5stars'].items():
                        self.standard_5stars[k] = [v, 0]
                else:
                    for name in config_data['standard_5stars']:
                        self.standard_5stars[name] = [-1, 0]
                
            # 如果配置中有 4stars，则更新
            if '4stars' in config_data:
                # 兼容字典或列表格式
                if isinstance(config_data['4stars'], dict):
                    for k, v in config_data['4stars'].items():
                        self._4stars[k] = [v, 0]
                else:
                    for name in config_data['4stars']:
                        self._4stars[name] = [-1, 0]

        self.pity_4star = 0

        # 武器池初始化
        self.pity_5star_weapon = self.initial_pity_weapon
        self.pity_4star_weapon = 0
        self.featured_weapon_count = 0

        self.total_afterglow_coral_count = self.initial_coral
        self.gained_afterglow_coral_count = 0
        self.total_oscillated_coral_count = 0

    @staticmethod
    def rate_5star(rate_number: int):
        if rate_number > 80: return 1.0
        return RATE_5_STAR[rate_number]

    @staticmethod
    def rate_4star(rate_number: int):
        if rate_number > 11: return 1.0
        return RATE_4_STAR[rate_number]

    def pull(self):
        """
        执行一次抽卡
        返回：
            tuple[str, int, int]
            --抽取到的角色或武器
            --本次抽取获得的余波珊瑚（大珊瑚）
            --本次抽取获取的残振珊瑚（小珊瑚）
        """
        # 本次抽到的角色或武器
        local_item = ''
        # 本次抽取获得的余波珊瑚（大珊瑚）
        local_obtained_afterglow_coral = 0
        # 本次抽取获取的残振珊瑚（小珊瑚）
        local_obtained_oscillated_coral = 0

        # 调用核心逻辑
        outcome, self.pity_5star, self.pity_4star, self.featured_5star_guaranteed, self.featured_4star_guaranteed = core_pull(
            self.pity_5star, self.pity_4star, self.featured_5star_guaranteed, self.featured_4star_guaranteed, 0
        )

        if outcome == OUTCOME_5_STAR_UP:
            # 抽到限定五星角色
            local_item = self.rate_up_5star
            if self.featured_5stars[local_item][0] < 7:
                # 限定五星角色未满链，链数和抽到数+1、获得15余波珊瑚（大珊瑚）
                self.featured_5stars[local_item][0] += 1
                self.featured_5stars[local_item][1] += 1
                local_obtained_afterglow_coral = 15
            else:
                # 限定五星角色已满链，链数不变，抽到数+1、获得40余波珊瑚（大珊瑚）
                self.featured_5stars[local_item][1] += 1
                local_obtained_afterglow_coral = 40

        elif outcome == OUTCOME_5_STAR_STANDARD:
            # 抽到常驻五星角色
            local_item = random.choice(list(self.standard_5stars.keys()))
            if self.standard_5stars[local_item][0] < 6:
                # 常驻五星角色未满链，链数和抽到数+1、获得45余波珊瑚（大珊瑚）
                self.standard_5stars[local_item][0] += 1
                self.standard_5stars[local_item][1] += 1
                local_obtained_afterglow_coral = 45
            else:
                # 常驻五星角色已满链，链数不变，抽到数+1、获得70余波珊瑚（大珊瑚）
                self.standard_5stars[local_item][1] += 1
                local_obtained_afterglow_coral = 70

        elif outcome == OUTCOME_4_STAR_UP:
            # 抽到概率up四星角色
            local_item = random.choice(self.rate_up_4stars)
            if self._4stars[local_item][0] < 6:
                # 四星角色未满链，链数和抽到数+1、获得3余波珊瑚（大珊瑚）
                self._4stars[local_item][0] += 1
                self._4stars[local_item][1] += 1
                local_obtained_afterglow_coral = 3
            else:
                # 四星角色已满链，链数不变，抽到数+1、获得8余波珊瑚（大珊瑚）
                self._4stars[local_item][1] += 1
                local_obtained_afterglow_coral = 8

        elif outcome == OUTCOME_4_STAR_OTHER:
            # 抽到非概率up四星角色或武器
            # 角色池非UP四星池：所有非UP四星角色 + 20把四星武器
            local_non_rate_up_4stars = [c for c in self._4stars if c not in self.rate_up_4stars]
            # 武器列表 (20把)
            weapon_pool_size = 20
            
            total_pool_size = len(local_non_rate_up_4stars) + weapon_pool_size
            r = random.randint(0, total_pool_size - 1)
            
            if r < len(local_non_rate_up_4stars):
                # 抽到角色
                local_item = local_non_rate_up_4stars[r]
                if self._4stars[local_item][0] < 6:
                    # 四星角色未满链，链数和抽到数+1、获得3余波珊瑚（大珊瑚）
                    self._4stars[local_item][0] += 1
                    self._4stars[local_item][1] += 1
                    local_obtained_afterglow_coral = 3
                else:
                    # 四星角色已满链，链数不变，抽到数+1、获得8余波珊瑚（大珊瑚）
                    self._4stars[local_item][1] += 1
                    local_obtained_afterglow_coral = 8
            else:
                # 抽到武器
                local_item = '4星武器'
                # 获得3余波珊瑚
                local_obtained_afterglow_coral = 3

        else:
            # 抽到三星武器
            local_item = '3星武器'
            # 三星武器计数+1
            self.weapon += 1
            # 抽到三星武器，获得15残振珊瑚（小珊瑚）
            local_obtained_oscillated_coral = 15

        return local_item, local_obtained_afterglow_coral, local_obtained_oscillated_coral

    def pull_weapon(self):
        """
        执行一次武器池抽卡
        返回：
            tuple[str, int, int]
            --抽取到的角色或武器
            --本次抽取获得的余波珊瑚（大珊瑚）
            --本次抽取获取的残振珊瑚（小珊瑚）
        """
        # 本次抽到的角色或武器
        local_item = ''
        # 本次抽取获得的余波珊瑚（大珊瑚）
        local_obtained_afterglow_coral = 0
        # 本次抽取获取的残振珊瑚（小珊瑚）
        local_obtained_oscillated_coral = 0

        # 调用核心逻辑 (banner_type=1)
        # 注意：武器池没有“大保底”概念，is_guaranteed传False即可，返回值忽略
        # 武器池使用四星保底逻辑
        outcome, self.pity_5star_weapon, self.pity_4star_weapon, _, self.featured_4star_weapon_guaranteed = core_pull(
            self.pity_5star_weapon, self.pity_4star_weapon, False, self.featured_4star_weapon_guaranteed, 1
        )

        if outcome == OUTCOME_5_STAR_UP:
            # 武器池五星必定是当期UP
            local_item = '当期专武'
            self.featured_weapon_count += 1
            # 假设重复获得五星武器给15余波珊瑚
            local_obtained_afterglow_coral = 15

        elif outcome == OUTCOME_4_STAR_UP:
            # UP 4星武器 (3把)
            local_item = 'UP 4星武器'
            local_obtained_afterglow_coral = 3

        elif outcome == OUTCOME_4_STAR_OTHER:
            # 非UP 4星 (所有角色 + 17把武器)
            all_chars = list(self._4stars.keys())
            weapon_pool_size = 17
            total_pool_size = len(all_chars) + weapon_pool_size
            
            r = random.randint(0, total_pool_size - 1)
            if r < len(all_chars):
                # 抽到角色
                local_item = all_chars[r]
                if self._4stars[local_item][0] < 6:
                    self._4stars[local_item][0] += 1
                    self._4stars[local_item][1] += 1
                    local_obtained_afterglow_coral = 3
                else:
                    self._4stars[local_item][1] += 1
                    local_obtained_afterglow_coral = 8
            else:
                # 抽到武器
                local_item = '4星武器'
                local_obtained_afterglow_coral = 3

        else:
            # 抽到三星武器
            local_item = '3星武器'
            # 三星武器计数+1
            self.weapon += 1
            # 抽到三星武器，获得15残振珊瑚（小珊瑚）
            local_obtained_oscillated_coral = 15

        return local_item, local_obtained_afterglow_coral, local_obtained_oscillated_coral

    def simulate_pulls(self, num_pulls: int, banner_type: str = 'character', verbose: bool = False):
        """
        模拟多次抽卡
        参数：
            num_pulls: 抽卡次数
            banner_type: 卡池类型 'character' 或 'weapon'
            verbose: 是否显示每次抽卡结果
        """
        # 执行指定次数抽卡
        for pull in range(num_pulls):
            # 总抽数更新
            self.pull_count += 1
            # 执行一次抽卡，获得抽取物、珊瑚
            if banner_type == 'weapon':
                local_item, local_obtained_afterglow_coral, local_obtained_oscillated_coral = self.pull_weapon()
            else:
                local_item, local_obtained_afterglow_coral, local_obtained_oscillated_coral = self.pull()
            
            self.total_afterglow_coral_count += local_obtained_afterglow_coral
            self.gained_afterglow_coral_count += local_obtained_afterglow_coral
            self.total_oscillated_coral_count += local_obtained_oscillated_coral

            if verbose:
                # verbose为True，输出抽卡结果
                if local_obtained_afterglow_coral > 8:
                    print(f'第{self.pull_count}抽：获得\033[33m{local_item:-^10}\033[0m和'
                          f'{local_obtained_afterglow_coral}余波珊瑚。')
                elif local_obtained_afterglow_coral > 0:
                    print(f'第{self.pull_count}抽：获得\033[35m{local_item:-^10}\033[0m和'
                          f'{local_obtained_afterglow_coral}余波珊瑚。')
                elif local_obtained_oscillated_coral > 0:
                    print(f'第{self.pull_count}抽：获得{local_item:-^10}和{local_obtained_oscillated_coral}残振珊瑚。')
                else:
                    # 正常情况下此种情况不存在
                    print(f'第{self.pull_count}抽：获得{local_item:-^10}。')

    def get_pity_info(self):
        """
        获取当前保底信息
        返回：
            dict: 包含保底抽数信息的字典
        """
        return {
            'pity_5star': self.pity_5star,
            'pity_4star': self.pity_4star,
            'featured_5star_guaranteed': self.featured_5star_guaranteed,
            'featured_4star_guaranteed': self.featured_4star_guaranteed
        }

    def get_character_stats(self):
        """
        获取所有角色统计信息
        返回：
            dict: 包含角色统计信息的字典
        """
        stats = {
            'featured_5star': {},
            'standard_5stars': {},
            'star_4s': {},
        }

        # 限定五星统计
        for char, (owned, obtained) in self.featured_5stars.items():
            if obtained > 0:
                stats['featured_5star'][char] = {
                    'owned': owned,
                    'obtained': obtained,
                }

        # 常驻五星统计
        for char, (owned, obtained) in self.standard_5stars.items():
            if obtained > 0:
                stats['standard_5stars'][char] = {
                    'owned': owned,
                    'obtained': obtained
                }

        # 四星统计
        for char, (owned, obtained) in self._4stars.items():
            if obtained > 0:
                stats['star_4s'][char] = {
                    'owned': owned,
                    'obtained': obtained
                }

        return stats

    def set_banner(self, featured_5star: str, featured_4stars: list):
        """
        设置当前卡池概率up角色
        参数:
            featured_5star: 限定5星角色名
            featured_4stars: UP四星角色列表
        """
        self.rate_up_5star = featured_5star
        self.rate_up_4stars = featured_4stars.copy()

        # 确保限定5星在字典中，否则将其加入角色池
        if featured_5star not in self.featured_5stars:
            self.featured_5stars[featured_5star] = [-1, 0]

        # 确保UP四星在字典中，否则将其加入角色池
        for char in featured_4stars:
            if char not in self._4stars:
                self._4stars[char] = [-1, 0]

@njit
def run_single_simulation(initial_guaranteed, initial_coral, initial_pity_5star, initial_pity_weapon, target_chain, target_weapon, initial_four_star_chains, initial_standard_chains):
    # 局部变量初始化
    pity_5 = initial_pity_5star
    pity_4 = 0
    
    # 角色池状态
    featured_chain = -1
    # standard_chains: 5个常驻角色
    standard_chains = initial_standard_chains.copy()
    # four_star_chains: 12个四星角色。前3个为UP，后9个为非UP
    four_star_chains = initial_four_star_chains.copy()
    
    # 统计四星获取情况
    four_star_obtained = np.zeros(12, dtype=np.int32)
    
    # 武器池状态
    pity_5_weapon = initial_pity_weapon
    pity_4_weapon = 0
    featured_weapon_count = 0
    is_4star_weapon_guaranteed = False
    
    # 资源
    coral = initial_coral
    oscillated_coral = 0
    gained_coral = 0
    pull_count = 0
    exchanged_num = 0
    
    is_guaranteed = initial_guaranteed
    is_4star_guaranteed = False
    
    # 1. 确保拥有角色 (0链)
    if target_chain >= 0 and featured_chain < 0:
        while featured_chain < 0:
            pull_count += 1
            outcome, pity_5, pity_4, is_guaranteed, is_4star_guaranteed = core_pull(pity_5, pity_4, is_guaranteed, is_4star_guaranteed, 0)
            
            if outcome == OUTCOME_5_STAR_UP:
                if featured_chain < 7: 
                    featured_chain += 1
                    c = 15
                else:
                    c = 40
                coral += c
                gained_coral += c
            elif outcome == OUTCOME_5_STAR_STANDARD:
                idx = int(random.random() * 5) # 0-4
                if standard_chains[idx] < 6:
                    standard_chains[idx] += 1
                    c = 45
                else:
                    c = 70
                coral += c
                gained_coral += c
            elif outcome == OUTCOME_4_STAR_UP:
                idx = int(random.random() * 3)
                four_star_obtained[idx] += 1
                if four_star_chains[idx] < 6:
                    four_star_chains[idx] += 1
                    c = 3
                else:
                    c = 8
                coral += c
                gained_coral += c
            elif outcome == OUTCOME_4_STAR_OTHER:
                # 9个非UP角色 + 20把武器 = 29
                r = random.random() * 29
                if r < 9:
                    # 角色 (假设前9个是非UP角色)
                    idx = 3 + int(r) # 3-11
                    four_star_obtained[idx] += 1
                    if four_star_chains[idx] < 6:
                        four_star_chains[idx] += 1
                        c = 3
                    else:
                        c = 8
                    coral += c
                    gained_coral += c
                else:
                    # 武器
                    c = 3
                    coral += c
                    gained_coral += c
            else: # 3 Star
                oscillated_coral += 15

    # 2. 抽武器
    while featured_weapon_count < target_weapon:
        pull_count += 1
        outcome, pity_5_weapon, pity_4_weapon, _, is_4star_weapon_guaranteed = core_pull(pity_5_weapon, pity_4_weapon, False, is_4star_weapon_guaranteed, 1)
        
        if outcome == OUTCOME_5_STAR_UP:
            featured_weapon_count += 1
            c = 15
            coral += c
            gained_coral += c
        elif outcome == OUTCOME_4_STAR_UP:
            # UP Weapon
            c = 3
            coral += c
            gained_coral += c
        elif outcome == OUTCOME_4_STAR_OTHER:
            # Other: 12 Chars + 17 Weapons = 29
            r = random.random() * 29
            if r < 12:
                # Character
                idx = int(r) # 0-11 (All chars)
                four_star_obtained[idx] += 1
                if four_star_chains[idx] < 6:
                    four_star_chains[idx] += 1
                    c = 3
                else:
                    c = 8
                coral += c
                gained_coral += c
            else:
                # Weapon
                c = 3
                coral += c
                gained_coral += c
        else: # 3 Star
            oscillated_coral += 15

    # 3. 补齐角色
    while True:
        if featured_chain >= 0:
            needed = target_chain - featured_chain
            if needed <= 0:
                break
            elif needed <= 2 and coral >= needed * 360:
                coral -= needed * 360
                exchanged_num += needed
                break
            elif needed > 0 and coral >= 360:
                coral -= 360
                exchanged_num += 1
                featured_chain += 1
                continue
        
        pull_count += 1
        outcome, pity_5, pity_4, is_guaranteed, is_4star_guaranteed = core_pull(pity_5, pity_4, is_guaranteed, is_4star_guaranteed, 0)
        
        if outcome == OUTCOME_5_STAR_UP:
            if featured_chain < 7:
                featured_chain += 1
                c = 15
            else:
                c = 40
            coral += c
            gained_coral += c
        elif outcome == OUTCOME_5_STAR_STANDARD:
            idx = int(random.random() * 5)
            if standard_chains[idx] < 6:
                standard_chains[idx] += 1
                c = 45
            else:
                c = 70
            coral += c
            gained_coral += c
        elif outcome == OUTCOME_4_STAR_UP:
            idx = int(random.random() * 3)
            four_star_obtained[idx] += 1
            if four_star_chains[idx] < 6:
                four_star_chains[idx] += 1
                c = 3
            else:
                c = 8
            coral += c
            gained_coral += c
        elif outcome == OUTCOME_4_STAR_OTHER:
            # 9个非UP角色 + 20把武器 = 29
            r = random.random() * 29
            if r < 9:
                # 角色 (假设前9个是非UP角色)
                idx = 3 + int(r) # 3-11
                four_star_obtained[idx] += 1
                if four_star_chains[idx] < 6:
                    four_star_chains[idx] += 1
                    c = 3
                else:
                    c = 8
                coral += c
                gained_coral += c
            else:
                # 武器
                c = 3
                coral += c
                gained_coral += c
        else: # 3 Star
            oscillated_coral += 15
    
    return pull_count, coral, oscillated_coral, gained_coral, exchanged_num, four_star_obtained, four_star_chains

@njit(parallel=True)
def run_simulations_parallel(n, initial_guaranteed, initial_coral, initial_pity_5star, initial_pity_weapon, target_chain, target_weapon, initial_four_star_chains, initial_standard_chains):
    # Pre-allocate arrays
    pulls_count_arr = np.zeros(n, dtype=np.int32)
    remaining_afterglow_arr = np.zeros(n, dtype=np.int32)
    remaining_oscillated_arr = np.zeros(n, dtype=np.int32)
    gained_afterglow_arr = np.zeros(n, dtype=np.int32)
    exchanged_chains_arr = np.zeros(n, dtype=np.int32)
    
    # 4-star stats arrays
    all_4star_obtained = np.zeros((n, 12), dtype=np.int32)
    all_4star_chains = np.zeros((n, 12), dtype=np.int32)

    for i in prange(n):
        res = run_single_simulation(initial_guaranteed, initial_coral, initial_pity_5star, initial_pity_weapon, target_chain, target_weapon, initial_four_star_chains, initial_standard_chains)
        pulls_count_arr[i] = res[0]
        remaining_afterglow_arr[i] = res[1]
        remaining_oscillated_arr[i] = res[2]
        gained_afterglow_arr[i] = res[3]
        exchanged_chains_arr[i] = res[4]
        all_4star_obtained[i] = res[5]
        all_4star_chains[i] = res[6]

    return pulls_count_arr, remaining_afterglow_arr, remaining_oscillated_arr, gained_afterglow_arr, exchanged_chains_arr, all_4star_obtained, all_4star_chains

if __name__ == '__main__':
    # 获取用户输入的目标链数
    try:
        target_chain_input = input("请输入想要获取的限定五星角色链数：")
        if not target_chain_input.strip():
            print("未输入，默认计算0链。")
            target_chain = 0
        else:
            target_chain = int(target_chain_input)
            if target_chain < 0 or target_chain > 6:
                print("输入错误，链数必须在0到6之间。")
                exit()
    except ValueError:
        print("输入错误，请输入整数。")
        exit()

    # 获取用户输入的目标专武数量
    try:
        target_weapon_input = input("请输入想要获取的限定五星专武数量：")
        if not target_weapon_input.strip():
            print("未输入，默认计算0把。")
            target_weapon = 0
        else:
            target_weapon = int(target_weapon_input)
            if target_weapon < 0 or target_weapon > 5:
                print("输入错误，数量必须在0到5之间。")
                exit()
    except ValueError:
        print("输入错误，请输入整数。")
        exit()

    # 获取用户输入的初始状态
    guaranteed_input = input("下一个五星是否大保底 (y/n): ")
    initial_guaranteed = True if guaranteed_input.strip().lower() == 'y' else False

    coral_input = input("目前有多少余波珊瑚: ")
    try:
        initial_coral = int(coral_input) if coral_input.strip() else 0
    except ValueError:
        print("输入错误，默认0。")
        initial_coral = 0

    pity_5star_input = input("角色池已垫抽数 (0-79，默认0): ")
    try:
        initial_pity_5star = int(pity_5star_input) if pity_5star_input.strip() else 0
        if initial_pity_5star < 0 or initial_pity_5star > 79:
            print("输入错误，默认0。")
            initial_pity_5star = 0
    except ValueError:
        print("输入错误，默认0。")
        initial_pity_5star = 0

    pity_weapon_input = input("武器池已垫抽数 (0-79，默认0): ")
    try:
        initial_pity_weapon = int(pity_weapon_input) if pity_weapon_input.strip() else 0
        if initial_pity_weapon < 0 or initial_pity_weapon > 79:
            print("输入错误，默认0。")
            initial_pity_weapon = 0
    except ValueError:
        print("输入错误，默认0。")
        initial_pity_weapon = 0

    # 获取初始四星状态 (从GachaSimulator类中读取)
    sim = GachaSimulator()
    initial_four_star_chains = np.zeros(12, dtype=np.int8)
    
    # 映射UP四星 (索引0-2)
    for i, char_name in enumerate(sim.rate_up_4stars):
        initial_four_star_chains[i] = sim._4stars[char_name][0]
        
    # 映射非UP四星 (索引3-11)
    non_up_chars = [c for c in sim._4stars if c not in sim.rate_up_4stars]
    for i, char_name in enumerate(non_up_chars):
        initial_four_star_chains[3 + i] = sim._4stars[char_name][0]

    # 获取初始常驻五星链数 (按字典序或固定顺序，需与 run_single_simulation 内部逻辑一致)
    # 在 run_single_simulation 中，我们用 idx = int(random.random() * 5) 随机选择
    # 因此这里只需要传入一个长度为5的数组，顺序并不重要，只要每次随机选一个即可
    # 但为了严谨，我们按 keys 的顺序传入
    initial_standard_chains = np.zeros(5, dtype=np.int8)
    for i, char_name in enumerate(sim.standard_5stars.keys()):
        initial_standard_chains[i] = sim.standard_5stars[char_name][0]

    # 模拟次数
    n = 1000000
    
    print(f"开始模拟抽取 {target_chain} 链角色 + {target_weapon} 把专武，模拟次数：{n}...")
    print(f"正在使用 Numba Parallel (OpenMP/TBB) 进行并行计算...")

    # 执行并行模拟
    # 注意：第一次运行会包含编译时间
    import time
    t0 = time.time()
    pulls_count_list, remaining_afterglow_list, remaining_oscillated_list, gained_afterglow_list, exchanged_chains_list, all_4star_obtained, all_4star_chains = run_simulations_parallel(
        n, initial_guaranteed, initial_coral, initial_pity_5star, initial_pity_weapon, target_chain, target_weapon, initial_four_star_chains, initial_standard_chains
    )
    t1 = time.time()
    print(f"模拟完成，耗时: {t1 - t0:.4f}s")

    average = np.mean(pulls_count_list)
    average_gained_afterglow = np.mean(gained_afterglow_list)
    average_exchanged_chains = np.mean(exchanged_chains_list)
    average_total_before_exchange = initial_coral + average_gained_afterglow

    # 计算四星统计数据
    avg_4star_obtained = np.mean(all_4star_obtained, axis=0)
    avg_4star_chains = np.mean(all_4star_chains, axis=0)
    
    # 重建四星名字列表以对应索引
    four_star_names = list(sim.rate_up_4stars) + [c for c in sim._4stars if c not in sim.rate_up_4stars]

    print("\n" + "="*45)
    print("四星角色统计 (平均值)")
    print(f"{'角色名':<10} {'初始链数':<10} {'获取数量':<10} {'最终链数':<10}")
    print("-" * 45)
    for i, name in enumerate(four_star_names):
        print(f"{name:<10} {initial_four_star_chains[i]:<10} {avg_4star_obtained[i]:<10.2f} {avg_4star_chains[i]:<10.2f}")
    print("="*45 + "\n")

    # 计算余波珊瑚众数
    vals_afterglow, counts_afterglow = np.unique(remaining_afterglow_list, return_counts=True)
    mode_afterglow = vals_afterglow[np.argmax(counts_afterglow)]

    # 计算残振珊瑚众数
    vals_oscillated, counts_oscillated = np.unique(remaining_oscillated_list, return_counts=True)
    mode_oscillated = vals_oscillated[np.argmax(counts_oscillated)]

    values, counts = np.unique(pulls_count_list, return_counts=True)
    
    # 计算频率峰值（众数）
    mode_pulls = values[np.argmax(counts)]
    # 计算最大值（模拟中的实际最大值）
    max_pulls = np.max(pulls_count_list)
    
    # 计算理论硬保底 (Theoretical Max)
    # 修正：基于“先抽1只角色 -> 抽满武器 -> 抽剩余角色”的最优策略计算
    
    # 检查四星是否全满命
    is_all_4star_full = np.all(initial_four_star_chains >= 6)
    coral_per_4star = 8 if is_all_4star_full else 3

    # 计算还需要抽取的角色数量
    # 假设初始为-1 (未拥有)
    current_copies = 0 
    target_copies = target_chain + 1
    needed_copies = max(0, target_copies - current_copies)
    
    # 武器池最大抽数
    max_weapon_pulls = target_weapon * 79 - initial_pity_weapon
    max_weapon_pulls = max(0, max_weapon_pulls)
    
    # 武器池最差珊瑚产出 (五星15 + 四星)
    weapon_coral = target_weapon * 15 + (max_weapon_pulls // 10) * coral_per_4star

    # 角色池最差情况分析
    first_char_pulls = 0
    first_char_coral = 0
    remaining_copies = needed_copies

    # 如果尚未拥有角色且需要抽取，则必须先抽 1 只解锁兑换
    if needed_copies > 0:
        # 计算第1只的代价
        if initial_guaranteed:
            first_char_pulls = 79 - initial_pity_5star
            first_char_coral = 15 # 必中
        else:
            first_char_pulls = 158 - initial_pity_5star
            first_char_coral = 60 # 歪(45) + 中(15)
        
        first_char_pulls = max(0, first_char_pulls)
        # 加上四星珊瑚
        first_char_coral += (first_char_pulls // 10) * coral_per_4star
        
        remaining_copies -= 1

    # 此时拥有的基础珊瑚 (初始 + 第1只产出 + 武器产出)
    base_coral = initial_coral + first_char_coral + weapon_coral
    
    # 剩余需要抽取的角色数: remaining_copies
    # 我们需要判断能兑换几个 (最多2个)
    exchangeable = 0
    max_ex = min(2, remaining_copies) # 最多换2个，且不能超过剩余需求
    
    # 辅助：计算后续 N 只角色的最大抽数和珊瑚
    def calc_rest_cost(n):
        if n <= 0: return 0, 0
        # 后续角色默认都是小保底开始 (最坏情况)
        # 每次 158 抽，产出 60 珊瑚
        p = n * 158
        c = n * 60 + (p // 10) * coral_per_4star
        return p, c

    if max_ex == 2:
        # 尝试换2个
        # 需要抽 remaining_copies - 2 个
        rem = remaining_copies - 2
        p, c = calc_rest_cost(rem)
        if base_coral + c >= 720:
            exchangeable = 2
        else:
            max_ex = 1
            
    if max_ex == 1 and exchangeable == 0:
        # 尝试换1个
        rem = remaining_copies - 1
        p, c = calc_rest_cost(rem)
        if base_coral + c >= 360:
            exchangeable = 1
            
    # 最终理论最大抽数
    # = 第一只抽数 + 武器抽数 + (剩余需抽数 * 158)
    theoretical_max = first_char_pulls + max_weapon_pulls + (remaining_copies - exchangeable) * 158

    # 计算累积概率
    cumulative_probabilities = np.cumsum(counts) / n * 100

    if go:
        # 使用 Plotly 绘制交互式 HTML 图表
        fig = go.Figure()

        # 1. 频率直方图 (左轴)
        fig.add_trace(go.Bar(
            x=values,
            y=counts,
            name='频率 (Frequency)',
            marker_color='skyblue',
            opacity=0.6,
            yaxis='y1',
            hovertemplate='抽数: %{x}<br>频率: %{y}<extra></extra>'
        ))

        # 2. 累积概率折线图 (右轴)
        fig.add_trace(go.Scatter(
            x=values,
            y=cumulative_probabilities,
            name='累积概率 (Cumulative %)',
            line=dict(color='red', width=3),
            yaxis='y2',
            hovertemplate='抽数: %{x}<br>累积概率: %{y:.2f}%<extra></extra>'
        ))

        # 3. 添加辅助线 (期望、众数、必得)
        # 定义一个添加美化标签的辅助函数
        def add_styled_annotation(x_val, text, color, y_pos, line_dash="dash", double_line=False):
            if double_line:
                # 双实线效果：底层宽线 + 顶层细白线 (利用遮罩模拟双线)
                fig.add_vline(x=x_val, line_width=5, line_dash="solid", line_color=color)
                fig.add_vline(x=x_val, line_width=2, line_dash="solid", line_color="white")
            else:
                fig.add_vline(x=x_val, line_width=3, line_dash=line_dash, line_color=color)

            fig.add_annotation(
                x=x_val, 
                y=y_pos, 
                yref="paper",
                text=f"<b>{text}</b>",
                showarrow=False,        # 不显示箭头
                font=dict(color=color, size=16, family="HarmonyOS Sans SC"),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=color,
                borderwidth=2,
                borderpad=6,
                xanchor="center"        # 居中对齐
            )

        # 策略：期望和众数通常非常接近，为了防止重叠，默认将它们错开高度
        # 期望放在第一层 (y=1.05)
        add_styled_annotation(average, f"期望: {average:.2f}", "orange", 1.05)
        
        # 众数放在第二层 (y=1.08)，这样无论如何缩放都不会重叠
        add_styled_annotation(mode_pulls, f"众数: {mode_pulls}", "green", 1.10)

        # 必得通常较远，可以放回第一层 (y=1.05)，与期望对齐
        # 显示模拟最大值
        add_styled_annotation(max_pulls, f"模拟最大抽数: {max_pulls}", "black", 1.05, line_dash="solid")
        
        # 如果模拟最大值没达到理论保底，额外显示理论保底
        if max_pulls <= theoretical_max:
             add_styled_annotation(theoretical_max, f"理论最大抽数: {theoretical_max}", "gray", 1.10, line_dash="solid")

        # 4. 更新布局配置
        fig.update_layout(
            font=dict(family="HarmonyOS Sans SC"), # 全局字体设置，涵盖大部分文本
            margin=dict(t=140), # 增加顶部边距以容纳两层标签
            title=dict(
                text=f'<b>鸣潮抽卡模拟分布</b> ({target_chain} 链角色 + {target_weapon} 把专武)',
                font=dict(size=24, family="HarmonyOS Sans SC")
            ),
            xaxis=dict(
                title=dict(text='抽数 (Pull Count)', font=dict(family="HarmonyOS Sans SC")),
                tickfont=dict(family="HarmonyOS Sans SC"), # 显式设置刻度字体
                gridcolor='rgba(0,0,0,0.1)'
            ),
            # 左轴：频率
            yaxis=dict(
                title=dict(text='频率 (Frequency)', font=dict(color='skyblue', family="HarmonyOS Sans SC")),
                tickfont=dict(color='skyblue', family="HarmonyOS Sans SC"), # 显式设置刻度字体
                showgrid=False,  # 隐藏频率网格，避免干扰
                fixedrange=True  # 锁定Y轴，只允许横向缩放
            ),
            # 右轴：累积概率
            yaxis2=dict(
                title=dict(text='累积概率 (%)', font=dict(color='red', family="HarmonyOS Sans SC")),
                tickfont=dict(color='red', family="HarmonyOS Sans SC"), # 显式设置刻度字体
                overlaying='y',
                side='right',
                range=[0, 105],
                showgrid=True,   # 显示概率网格
                gridcolor='rgba(255, 0, 0, 0.1)', # 红色淡网格
                dtick=10,        # 每10%一条线
                fixedrange=True  # 锁定Y轴，只允许横向缩放
            ),
            # 交互设置
            hovermode="x unified",  # 鼠标悬停时显示该x轴上所有数据
            template="plotly_white", # 简洁白底风格
            dragmode='pan',         # 默认鼠标操作为平移，体验更丝滑
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                font=dict(family="HarmonyOS Sans SC")
            ),
            # 优化悬停标签样式
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.95)",
                font_size=14,
                font_family="HarmonyOS Sans SC"
            )
        )

        # config 配置：开启滚轮缩放，隐藏 Plotly logo，开启响应式
        fig.show(config={
            'scrollZoom': True, 
            'displaylogo': False, 
            'responsive': True,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'] # 移除不常用的选择工具，保持界面简洁
        })
    else:
        print("\n[警告] 未安装 plotly，无法生成图表。请运行 pip install plotly 安装。")

    print("-----------------------------------------------------------")
    print(f'共经过 {n} 次模拟，获得当期 {target_chain} 链限定五星角色 + {target_weapon} 把专武所需的抽数统计：')
    print(f'期望抽数：{average:.2f} (约 {int(average * 160)} 星声)')
    print(f'众数抽数：{mode_pulls} (约 {mode_pulls * 160} 星声)')
    print(f'模拟最大抽数：{max_pulls} (约 {max_pulls * 160} 星声)')
    print(f'理论最大抽数：{theoretical_max} (约 {theoretical_max * 160} 星声)')
    print(f'（注：以上统计已包含余波珊瑚换取限定角色共鸣链，最多换取2个）')
    print("-----------------------------------------------------------")