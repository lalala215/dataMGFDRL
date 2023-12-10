#!/usr/bin/env python3
#
#  tcl_env.py
#  TCL environment for RL algorithms
#
# Author: Taha Nakabi
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import gym
# Trying out if this works for others. from gym import spaces had some issues
import gym.spaces as spaces

import math

# Default parameters for 
# default TCL environment.
# From Taha's code
DEFAULT_ITERATIONS = 24     # 定义迭代次数
DEFAULT_NUM_TCLS = 100
DEFAULT_NUM_LOADS = 150     # 定义负荷数量
# Load up default prices and 
# temperatures (from Taha's CSV)
default_data = np.load("default_price_and_temperatures.npy")    # 导入价格与温度
DEFAULT_PRICES = default_data[:,0]
DEFAULT_TEMPERATURS = default_data[:,1]
BASE_LOAD = np.array([2.0,2.0,2.0,2.0,3.4,4.0,6.0,5.5,6.0,5.5,4.0,3.3,4.1,3.3,4.1,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0])
# https://austinenergy.com/ae/residential/rates/residential-electric-rates-and-line-items
PRICE_TIERS = np.array([2.8,5.8,7.8,9.3,10.81])     # 价格等级

HIGH_PRICE_PENALTY = 2.0
FIXED_COST = 0
QUADRATIC_PRICE = .025

# Default Tmin and Tmax in TCLs
TCL_TMIN = 19
TCL_TMAX = 25   # 定义温度上下限
TCL_PENALTY=0.1     # TCL越限惩罚
MAX_R = 1
MAX_GENERATION = 120
SOCS_RENDER=[]
LOADS_RENDER =[]
BATTERY_RENDER = []
PRICE_RENDER = []
ENERGY_SOLD_RENDER = []
ENERGY_BOUGHT_RENDER = []
GRID_PRICES_RENDER = []
ENERGY_GENERATED_RENDER = []
TCL_CONTROL_RENDER=[]
TCL_CONSUMPTION_RENDER=[]




class TCL:
    """ 
    Simulates an invidual TCL
    """
    def __init__(self, ca, cm, q, P, Tmin=TCL_TMIN, Tmax=TCL_TMAX):
        self.ca = ca        # 空气热量赋值
        self.cm = cm        # 建筑材料热量赋值，
        self.q = q          # 建筑物内部热量赋值
        self.P = P          # 额定共六百赋值
        self.Tmin = Tmin    # 温度上下限赋值
        self.Tmax = Tmax

        # Added for clarity
        self.u = 0          # 控制信号初始化

    def set_T(self, T, Tm):
        self.T = T          # 当前温度赋值
        self.Tm = Tm        # 不可观测的建筑质量温度赋值

    def control(self, ui=0):    # 根据当前温度给TCL控制信号赋值
        # control TCL using u with respect to the backup controller
        if self.T < self.Tmin:
            self.u = 1
        elif self.Tmin<self.T<self.Tmax:
            self.u = ui
        else:
            self.u = 0

    def update_state(self, T0): # T0为室外温度
        # update the indoor and mass temperatures according to (22)
        for _ in range(5):      # 每个TCL的温度动态二阶模型
            self.T += self.ca * (T0 - self.T) + self.cm * (self.Tm - self.T) + self.P * self.u +self.q
            self.Tm += self.cm*(self.T - self.Tm)
            if self.T>=self.Tmax:
                 break

    """ 
    @property allows us to write "tcl.SoC", and it will
    run this function to get the value
    """
    @property
    def SoC(self):  #定义TCL的SoC
        return (self.T-self.Tmin)/(self.Tmax-self.Tmin)

class Battery:
    # Simulates the battery system of the microGrid
    def __init__(self, capacity, useD, dissipation, lossC, rateC, maxDD, chargeE, tmax):
        self.capacity = capacity    # 电池满容量
        self.useD = useD    # 有效放电系数
        self.dissipation = dissipation  # 电池耗散系数（电池储存的能量会逐渐耗散）
        self.lossC = lossC  # 充电损失
        self.rateC = rateC  # 充电比率
        self.maxDD = maxDD  # 单位时间电池传输最大功率
        self.tmax= tmax     # 最大充电时长
        self.chargeE = chargeE      # 给电池充电的能量供应量
        self.RC = 0     # 已存储容量
        self.ct = 0     # 充电步骤

    def charge(self, E):    # 能量过剩时，把能量储存进电池中
        empty = self.capacity-self.RC
        if empty <= 0:      # 若电池没余量，则不充电，返回需要存储的能量
            return E
        else:               # 若电池有余量，则充电
            self.RC += self.rateC*E
            leftover = self.RC - self.capacity      # 计算多余未充入电池的容量
            self.RC = min(self.capacity,self.RC)
            return max(leftover,0)  # 返回多余能量


    def supply(self, E):    # 能量不足时，则需要电池提供能量
        remaining = self.RC
        self.RC-= E*self.useD
        self.RC = max(self.RC,0)
        return min(E, remaining)    # 返回电池能够提供的能量

    def dissipate(self):    # 电池储存能量耗散量
        self.RC = self.RC * math.exp(- self.dissipation)

    @property
    def SoC(self):  # 计算电池SoC
        return self.RC/self.capacity

class Grid:
    def __init__(self):
        down_reg_df=pd.read_csv("down_regulation.csv")  # 下调价格数据
        up_reg_df = pd.read_csv("up_regulation.csv")    # 上调价格数据
        down_reg = np.array(down_reg_df.iloc[:,-1])/10  # 数据处理
        up_reg = np.array(up_reg_df.iloc[:, -1])/10
        self.buy_prices = down_reg
        self.sell_prices = up_reg
        self.time = 0

    def sell(self, E):  # 售电
        return self.sell_prices[self.time]*E

    def buy(self, E):   # 购电
        return -self.buy_prices[self.time]*E - QUADRATIC_PRICE*E**2 - FIXED_COST
    #
    # def get_price(self,time):
    #     return self.prices[time]

    def set_time(self,time):
        self.time = time


class Generation:
    def __init__(self, max_capacity=None):
        power_df = pd.read_csv("wind_generation.csv")   # 风力发电数据
        self.power = np.array(power_df.iloc[:,-1])
        self.max_capacity = np.max(self.power)  # 取最大风力发电数值

    def current_generation(self,time):  # 返回瞬时风力发电量
        # We consider that we have 2 sources of power a constant source and a variable source
        return  self.power[time]


class Load:
    def __init__(self, price_sens, base_load, max_v_load):  # 住宅价格响应负荷参数定义
        self.price_sens = price_sens    # 敏感因子
        self.base_load = base_load      # 基本负荷
        self.max_v_load = max_v_load    # 最大容忍参数
        self.response = 0

    def react(self, price_tier):    # 对于价格调整动作的响应
        self.response = self.price_sens*(price_tier-2)
        if self.response > 0 and self.price_sens > 0.1:
            self.price_sens-= 0.1

    def load(self, time_day):       # 计算负荷
        # print(self.response)
        return max(self.base_load[time_day] - self.max_v_load*self.response,0)



class TCLEnv(gym.Env):
    def __init__(self, **kwargs):
        """
        Arguments:
            iterations: Number of iterations to run
            num_tcls: Number of TCLs to create in cluster
            prices: Numpy 1D array of prices at different times
            temperatures : Numpy 1D array of temperatures at different times
        """

        # Get number of iterations and TCLs from the 
        # parameters (we have to define it through kwargs because 
        # of how Gym works...)
        # 微电网各参数赋值
        self.iterations = kwargs.get("iterations", DEFAULT_ITERATIONS)
        self.num_tcls = kwargs.get("num_tcls", DEFAULT_NUM_TCLS)
        self.num_loads = kwargs.get("num_loads", DEFAULT_NUM_LOADS)
        self.prices = kwargs.get("prices", DEFAULT_PRICES)
        self.temperatures = kwargs.get("temperatures", DEFAULT_TEMPERATURS)
        self.base_load = kwargs.get("base_load", BASE_LOAD)
        self.price_tiers = kwargs.get("price_tiers", PRICE_TIERS)

        # The current day: pick randomly
        self.day = random.randint(0,10)
        # self.day = 55
        # The current timestep
        self.time_step = 0



        # The cluster of TCLs to be controlled.
        # These will be created in reset()
        self.tcls_parameters = []   # 创建TCL
        self.tcls = []
        # The cluster of loads.
        # These will be created in reset()
        self.loads_parameters = []  # 创建住宅价格负载
        self.loads = []

        self.generation = Generation(MAX_GENERATION)    # 初始化风力发电
        self.grid = Grid()          # 初始化主网

        for i in range(self.num_tcls):      # 各TCL参数创建
            self.tcls_parameters.append(self._create_tcl_parameters())

        for i in range(self.num_loads):     # 各负荷参数创建
            self.loads_parameters.append(self._create_load_parameters())

        self.action_space = spaces.Box(low=0, high=1, dtype=np.float32,
                    shape=(13,))    # 动作空间初始化
        
        # Observations: A vector of TCLs SoCs + loads +battery soc+ power generation + price + temperature + time of day
        self.observation_space = spaces.Box(low=-100, high=100, dtype=np.float32, 
                    shape=(self.num_tcls  + 6,))    # 状态空间初始化


    def _create_tcl_parameters(self):   # TCL参数的赋值
        """
                Initialize one TCL randomly with given T_0,
                and return it. Copy/paste from Taha's code
                """
        # Hardcoded initialization values to create
        # bunch of different TCLs
        ca = random.normalvariate(0.004, 0.0008)
        cm = random.normalvariate(0.2, 0.004)
        q = random.normalvariate(0, 0.01)
        P = random.normalvariate(1.5, 0.01)
        return [ca,cm,q,P]

    def _create_tcl(self,ca ,cm ,q ,P, initial_temperature):    # 根据已有参数创建，计算其他参数
        tcl= TCL(ca,cm,q,P)
        tcl.set_T(initial_temperature,initial_temperature)
        return tcl  # 从而返回已创建的TCL类
    def _create_load_parameters(self):  # 负荷各参数赋值

        """
        Initialize one load randomly,
        and return it.
        """
        # Hardcoded initialization values to create
        # bunch of different loads

        price_sensitivity= random.normalvariate(0.5, 0.3)
        max_v_load = random.normalvariate(3.0, 1.0)
        return [price_sensitivity,max_v_load]

    def _create_load(self,price_sensitivity,max_v_load):
        load = Load(price_sensitivity,base_load=self.base_load, max_v_load=max_v_load)
        return load     # 返回已创建的负荷类

    def _create_battery(self):  # 电池参数赋值
        """
        Initialize one battery
        """
        battery = Battery(capacity = 400.0, useD=0.9, dissipation=0.001, lossC=0.15, rateC=0.9, maxDD=10, chargeE=10, tmax=5)
        return battery  # 返回已创建的电池类

    def _build_state(self):     # 状态空间创建
        """ 
        用一个向量表示状态
        返回：
            状态: 包括 TCLs的 SoC，负荷， 电池SoC， 风力发电功率， 温度， 电价， 一天中的时间段
        """
        # SoCs of all TCLs binned + current temperature + current price + time of day (hour)
        socs = np.array([tcl.SoC for tcl in self.tcls])     # TCLs的SoC赋值
        # Scaling between -1 and 1
        socs = (socs+np.ones(shape=socs.shape)*4)/(1+4)     # 归一化SoC

        # 当前时间各负荷值赋值
        loads = sum([l.load(self.time_step) for l in self.loads])
        # 负荷数据格式处理
        loads = (loads-(min(BASE_LOAD)+2)*DEFAULT_NUM_LOADS)/((max(BASE_LOAD)+4-min(BASE_LOAD)-2)*DEFAULT_NUM_LOADS)
        # print(loads)
        current_generation = self.generation.current_generation(self.day+self.time_step)    # 当前时间段风力发电
        current_generation /= self.generation.max_capacity  # 取最大发电量
        temperature = self.temperatures[self.day+self.time_step]    # 温度赋值
        temperature = (temperature-min(self.temperatures))/(max(self.temperatures)-min(self.temperatures))  # 温度格式化
        price = self.grid.buy_prices[self.day+self.time_step]   # 购电价格赋值
        price = (price - min(self.grid.buy_prices)) / (max(self.grid.buy_prices) - min(self.grid.buy_prices))   # 电价格式化
        time_step = self.time_step/24
        state = np.concatenate((socs, [loads,self.battery.SoC, current_generation,
                         temperature,
                         price,
                         time_step ]))  # 状态空间创建完成
        return state

    def _build_info(self):  # 温度和电价预测
        """
        Return dictionary of misc. infos to be given per state.
        Here this means providing forecasts of future
        prices and temperatures (next 24h)
        """
        temp_forecast = np.array(self.temperatures[self.time_step+1:self.time_step+25])
        price_forecast = np.array(self.prices[self.time_step+1:self.time_step+25])
        return {"temperature_forecast": temp_forecast,
                "price_forecast": price_forecast,
                "forecast_times": np.arange(0,self.iterations)}

    
    def _compute_tcl_power(self):   # 计算TCLs组所需总能量
        """
        Return the total power consumption of all TCLs
        """
        return sum([tcl.u*tcl.P for tcl in self.tcls])

    def step(self, action):
        """ 
        Arguments:
            action: A scalar float. 
        
        Returns:
            state: Current state
            reward: How much reward was obtained on last action
            terminal: Boolean on if the game ended (maximum number of iterations)
            info: None (not used here)
        """

        self.grid.set_time(self.day+self.time_step)
        reward = 0
        # Update state of TCLs according to action
        # 动作空间由四个动作组成，包括TCL动作、定价动作、微网能量不足动作、 微网能量过剩动作
        tcl_action = action[0]
        price_action = action[1]
        energy_deficiency_action = action[2]
        energy_excess_action = action[3]
        # 获取风力发电能量值
        available_energy = self.generation.current_generation(self.day+self.time_step)
        # Energy rate
        # self.eRate = available_energy/self.generation.max_capacity

        # print("Generated power: ", available_energy)
        # We implement the pricing action and we calculate the total load in response to the price
        for load in self.loads:
            load.react(price_action)
        total_loads = sum([l.load(self.time_step) for l in self.loads])     # 计算价格动作实施后的总负荷值
        # print("Total loads",total_loads)
        # We fulfilled the load with the available energy.
        available_energy -= total_loads     # 风力发电量去掉供给负荷能量后的可用能量值
        # We calculate the return based on the sale price.
        self.sale_price = self.price_tiers[price_action]
        # We increment the reward by the amount of return
        # Division by 100 to transform from cents to euros
        reward += total_loads*self.sale_price/100   # 奖励中加入从负荷处所得电能利润
        # 交易过高价格的惩罚
        self.high_price += price_action
        # 根据优先级把能量分配给TCLs组中的各个TCL
        sortedTCLs = sorted(self.tcls, key=lambda x: x.SoC)
        # print(tcl_action)
        control = tcl_action*50.0   # TCLs分配总量
        self.control = control
        # TCL分配过程如下
        for tcl in sortedTCLs:
            if control>0:
                tcl.control(1)
                control-= tcl.P * tcl.u
            else:
                tcl.control(0)
            tcl.update_state(self.temperatures[self.day+self.time_step])
            # if tcl.SoC >1 :
            #     reward -= abs((tcl.SoC-1) * reward*TCL_PENALTY)
            # if  tcl.SoC<0:
            #     reward += tcl.SoC * abs(reward*TCL_PENALTY)

        available_energy -= self._compute_tcl_power()   # 可用能量再减去TCLs所需总能量
        # control_error = self.sale_price*(self.control-self._compute_tcl_power())**2
        reward += self._compute_tcl_power()*self.sale_price/100     # 奖励中加入从TCLs所得利润
        if available_energy>0:  # 可用能量还有剩余
            if energy_excess_action:    # 判断是否执行能量过剩动作
                available_energy = self.battery.charge(available_energy)    # 将过剩能量充给电池
                reward += self.grid.sell(available_energy)/100  # 奖励中加入出售给主网所得利润
            else:
                reward += self.grid.sell(available_energy)/100
            self.energy_sold = available_energy
            self.energy_bought = 0

        else:   # 可用能量不足以提供给TCL
            if energy_deficiency_action:    # 判断是否执行能量不足动作
                available_energy += self.battery.supply(-available_energy)

            self.energy_bought = -available_energy  # 向主网购买所缺电能
            reward += self.grid.buy(self.energy_bought)/100 # 不知道为什么这里reward还是加
            self.energy_sold = 0

        # Proceed to next timestep.
        self.time_step += 1
        # Build up the representation of the current state (in the next timestep)
        state = self._build_state() # 获得下一状态
        terminal = self.time_step == self.iterations-1
        if self.high_price > 4 * self.iterations / 2:   # 执行过高价格交易惩罚
            # Penalize high prices
            reward -= abs(reward * HIGH_PRICE_PENALTY * (self.high_price - 4 * self.iterations / 2))
        if terminal:    # 执行电池充电奖励
            # reward if battery is charged
            reward += abs(reward*self.battery.SoC / 4)
        info = self._build_info()

        return state, reward, terminal, info

    def reset(self):    # 重置整个环境，返回初始化的环境和状态空间
        """
        Create new TCLs, and return initial state.
        Note: Overrides previous TCLs
        """
        # 初始化所有参数
        self.day = random.randint(0,10)
        # self.day = 5
        print("Day:",self.day)
        self.time_step = 0
        self.battery = self._create_battery()
        self.energy_sold = 0
        self.energy_bought = 0
        self.energy_generated = 0
        self.control=0
        self.sale_price = PRICE_TIERS[2]
        self.high_price = 0
        self.tcls.clear()
        initial_tcl_temperature = random.normalvariate(12, 5)

        for i in range(self.num_tcls):
            parameters = self.tcls_parameters[i]

            self.tcls.append(self._create_tcl(parameters[0],parameters[1],parameters[2],parameters[3],initial_tcl_temperature))

        self.loads.clear()
        for i in range(self.num_loads):
            parameters = self.loads_parameters[i]
            self.loads.append(self._create_load(parameters[0],parameters[1]))
        self.battery = self._create_battery()
        return self._build_state()

    def render(self, s):    # 画图程序
        SOCS_RENDER.append([tcl.SoC for tcl in self.tcls])
        LOADS_RENDER.append([l.load(self.time_step) for l in self.loads])
        PRICE_RENDER.append(self.sale_price)
        BATTERY_RENDER.append(self.battery.SoC)
        ENERGY_GENERATED_RENDER.append(self.generation.current_generation(self.day+self.time_step))
        ENERGY_SOLD_RENDER.append(self.energy_sold)
        ENERGY_BOUGHT_RENDER.append(self.energy_bought)
        GRID_PRICES_RENDER.append(self.grid.buy_prices[self.day+self.time_step])
        TCL_CONTROL_RENDER.append(self.control)
        TCL_CONSUMPTION_RENDER.append(self._compute_tcl_power())
        if self.time_step==self.iterations-1:
            fig=pyplot.figure()
            ax1 = fig.add_subplot(3,3,1)
            ax1.boxplot(np.array(SOCS_RENDER).T)
            ax1.set_title("TCLs SOCs")
            ax1.set_xlabel("Time (h)")
            ax1.set_ylabel("SOC")

            ax2 = fig.add_subplot(3, 3, 2)
            ax2.boxplot(np.array(LOADS_RENDER).T)
            ax2.set_title("LOADS")
            ax2.set_xlabel("Time (h)")
            ax2.set_ylabel("HOURLY LOADS")

            ax3 = fig.add_subplot(3, 3, 3)
            ax3.plot(PRICE_RENDER)
            ax3.set_title("SALE PRICES")
            ax3.set_xlabel("Time (h)")
            ax3.set_ylabel("HOURLY PRICES")

            ax4 = fig.add_subplot(3, 3, 4)
            ax4.plot(np.array(BATTERY_RENDER))
            ax4.set_title("BATTERY SOC")
            ax4.set_xlabel("Time (h)")
            ax4.set_ylabel("BATTERY SOC")

            ax4 = fig.add_subplot(3, 3, 5)
            ax4.plot(np.array(ENERGY_GENERATED_RENDER))
            ax4.set_title("ENERGY_GENERATED")
            ax4.set_xlabel("Time (h)")
            ax4.set_ylabel("ENERGY_GENERATED")

            ax4 = fig.add_subplot(3, 3, 6)
            ax4.plot(np.array(ENERGY_SOLD_RENDER))
            ax4.set_title("ENERGY_SOLD")
            ax4.set_xlabel("Time (h)")
            ax4.set_ylabel("ENERGY_SOLD")

            ax4 = fig.add_subplot(3, 3, 7)
            ax4.plot(np.array(ENERGY_BOUGHT_RENDER))
            ax4.set_title("ENERGY_BOUGHT")
            ax4.set_xlabel("Time (h)")
            ax4.set_ylabel("ENERGY_BOUGHT")

            ax4 = fig.add_subplot(3, 3, 8)
            ax4.plot(np.array(GRID_PRICES_RENDER))
            ax4.set_title("GRID_PRICES")
            ax4.set_xlabel("Time (h)")
            ax4.set_ylabel("GRID_PRICES_RENDER")

            ax4 = fig.add_subplot(3, 3, 9)
            ax4.bar(x=np.array(np.arange(self.iterations)),height=TCL_CONTROL_RENDER,width=0.2)
            ax4.bar(x=np.array(np.arange(self.iterations))+0.2,height=TCL_CONSUMPTION_RENDER,width=0.2)
            ax4.set_title("TCL_CONTROL VS TCL_CONSUMPTION")
            ax4.set_xlabel("Time (h)")
            ax4.set_ylabel("kW")

            pyplot.show()

            SOCS_RENDER.clear()
            LOADS_RENDER.clear()
            PRICE_RENDER.clear()
            BATTERY_RENDER.clear()
            GRID_PRICES_RENDER.clear()
            ENERGY_BOUGHT_RENDER.clear()
            ENERGY_SOLD_RENDER.clear()
            ENERGY_GENERATED_RENDER.clear()
            TCL_CONTROL_RENDER.clear()
            TCL_CONSUMPTION_RENDER.clear()


    def close(self):
        """ 
        Nothing to be done here, but has to be defined 
        """
        return

    def seed(self, seed):
        """
        Set the random seed for consistent experiments
        """
        random.seed(seed)
        np.random.seed(seed)
        
if __name__ == '__main__':
    # Testing the environment
    from matplotlib import pyplot
    from tqdm import tqdm
    env = TCLEnv()
    env.seed(1)
    #
    states = []
    rewards = []
    state = env.reset()
    states.append(state)
    actions = []
    #
    for i in tqdm(range(100)):
        action = env.action_space.sample()
        # print(action)
        actions.append(action)
        state, reward, terminal, _ = env.step(action)
        print(reward)
        states.append(state)
        rewards.append(reward)
        if terminal:
            break

    # Plot the TCL SoCs 
    states = np.array(rewards)
    pyplot.plot(rewards)
    pyplot.title("rewards")
    pyplot.xlabel("Time")
    pyplot.ylabel("rewards")
    pyplot.show()

    # battery = Battery(capacity = 75.0, useD = 0.8, dissipation = 0.001, lossC = 0.15, rateC = 0.5, maxDD = 10, chargeE=10, tmax = 5)
    # RCs = []
    # for itr in range(5):
    #     RCs.append(battery.SoC)
    #     battery.charge()
    # for itr in range(3):
    #     RCs.append(battery.SoC)
    #     battery.dissipate()
    # for itr in range(5):
    #     RCs.append(battery.SoC)
    #     battery.supply(5)
    # for itr in range(5):
    #     RCs.append(battery.SoC)
    #     battery.charge()
    # for itr in range(10):
    #     RCs.append(battery.SoC)
    #     battery.supply(20)
    #
    # pyplot.plot(RCs)
    # pyplot.show()
