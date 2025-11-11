# -*- coding: utf-8 -*-
"""
CFD的简单模拟 (Simple CFD Mock)
替代 6SigmaDCX.
(Replaces 6SigmaDCX)
"""

import numpy as np
import config


class SimpleCFD:
    """
    一个简化的热力学模型 (A simplified thermal model)
    输入: 16个机架的功耗 (Input: 16 rack powers)
    输出: 16个机架的出风温度 (Output: 16 rack outlet temperatures)
    """

    def __init__(self):
        self.num_racks = config.NUM_RACKS

        # 论文 3.2.2: "平均送风温度设置在20℃"
        # (Paper 3.2.2: "Average supply temp set to 20C")
        self.supply_temp = config.ACU_SUPPLY_TEMP

        # 这是一个物理系数 (Cubic Feet per Minute / Watt)
        # 它代表1W功耗能使多少空气升高1°C
        # 这是一个需要调优的模拟参数
        # (This is a tuning parameter for the simulation)
        self.thermal_coeff = 0.015

        # 空气混合系数 (0=完美混合, 1=无混合)
        # (Air mixing factor (0=perfect, 1=no mixing))
        self.mixing_factor = 0.7

        # 状态: 机架背部出风温度 T_o (State: Rack outlet temp T_o)
        # 初始化为合理的空闲温度
        # (Initialize to a reasonable idle temp)
        self.rack_outlet_temps = np.full(
            self.num_racks,
            self.supply_temp + 5.0
        )

    def update_rack_temps(self, rack_powers):
        """
        根据机架功耗更新温度
        (Update temperatures based on rack powers)

        rack_powers: shape (16,)
        """

        # 1. 计算每个机架的理想温升 (Delta T)
        # (Calculate ideal Delta T for each rack)
        # Delta T = Power * Thermal_Coefficient
        avg_power_per_rack = rack_powers / config.SERVERS_PER_RACK
        delta_t = avg_power_per_rack * self.thermal_coeff

        # 2. 计算理想出风口温度 (Calculate ideal outlet temp)
        ideal_outlet_temp = self.supply_temp + delta_t

        # 3. 模拟空气混合
        # (Simulate air mixing)
        # 新温度是自身理想温度和机房平均温度的加权平均
        # (New temp is a mix of its own ideal temp and the room avg temp)
        avg_room_temp = np.mean(self.rack_outlet_temps)
        mixed_temp = (ideal_outlet_temp * self.mixing_factor) + \
                     (avg_room_temp * (1.0 - self.mixing_factor))

        # 4. 模拟热惯性 (时间平滑)
        # (Simulate thermal inertia (temporal smoothing))
        self.rack_outlet_temps = (self.rack_outlet_temps * 0.9) + (mixed_temp * 0.1)

        return self.rack_outlet_temps

    def get_temperatures(self):
        """ 返回 T_o 向量 (Return T_o vector) """
        return self.rack_outlet_temps