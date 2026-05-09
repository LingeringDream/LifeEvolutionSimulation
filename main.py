import os
import matplotlib.pyplot as plt
import requests
import json
import time
import socket
import subprocess
import re
from scipy.signal import convolve2d
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# 基础配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False







def get_ollama_url():
    ollama_host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    host, port = ollama_host.split(":")
    return f"http://{host}:{port}/v1", host, int(port)


class PlanetEnvironmentLoader:
    """星球环境数据导入模块：负责从 Excel 解析物理特征并初始化模拟环境"""

    def __init__(self, target_size=50):
        self.target_size = target_size
        self.physics = {}  # 存储单值物理属性 (质量, 重力, 倾角等)
        self.composition = {}  # 存储大气成分

    def load_planet_config(self, file_path):
        """主导入函数"""
        try:
            with pd.ExcelFile(file_path) as xls:
                # 1. 导入物理特征 (单值属性)
                df_phys = pd.read_excel(xls, 'Physical', index_col=0)
                self.physics = df_phys.to_dict()['Value']

                # 2. 导入大气成分
                df_atmos = pd.read_excel(xls, 'Atmosphere', index_col=0)
                self.composition = df_atmos.to_dict()['Ratio']

                # 3. 导入空间分布数据 (如温度场/资源分布)
                # 如果没有这些表，则生成符合物理特征的初始场
                temp_grid = self._load_or_generate_grid(xls, 'Temperature', self._calc_base_temp())
                res_grid = self._load_or_generate_grid(xls, 'Resources', 1.0)

                print(f"--- 星球配置载入成功 ---")
                print(f"目标星球: {self.physics.get('Name', '未知')}")
                print(f"表面重力: {self.physics.get('Gravity', 9.8)} m/s²")
                print(f"转轴倾角: {self.physics.get('Axial_Tilt', 0)}°")

                return {
                    "physics": self.physics,
                    "composition": self.composition,
                    "grids": {"temp": temp_grid, "res": res_grid}
                }

        except Exception as e:
            print(f"配置导入失败: {e}，正在生成随机类地行星参数...")
            return self._generate_fallback_data()

    def _load_or_generate_grid(self, xls, sheet_name, default_val):
        """从Excel读取网格数据，若无则按默认值填充"""
        try:
            data = pd.read_excel(xls, sheet_name, header=None).values
            return self._resize_data(data)
        except:
            return np.full((self.target_size, self.target_size), default_val)

    def _resize_data(self, data):
        """插值缩放逻辑"""
        h, w = data.shape
        x = np.linspace(0, 1, h)
        y = np.linspace(0, 1, w)
        interp_func = RegularGridInterpolator((x, y), data)

        new_coords = np.linspace(0, 1, self.target_size)
        mesh_x, mesh_y = np.meshgrid(new_coords, new_coords, indexing='ij')
        return interp_func((mesh_x, mesh_y))

    def _calc_base_temp(self):
        """根据反照率(Albedo)和距离粗略计算基础温度 (物理公式简化)"""
        # 简化版：距离恒星越近温度越高
        dist = self.physics.get('Orbital_Distance', 1.0)  # AU
        albedo = self.physics.get('Albedo', 0.3)
        return 288 / np.sqrt(dist) * (1 - albedo) ** 0.25 - 273.15  # 摄氏度

    def _generate_fallback_data(self):
        """默认的回退数据生成器"""
        self.physics = {"Gravity": 9.8, "Axial_Tilt": 23.5, "Mass": 5.97e24}
        temp_grid = np.full((self.target_size, self.target_size), 15.0)
        res_grid = np.random.rand(self.target_size, self.target_size)
        return {"physics": self.physics, "grids": {"temp": temp_grid, "res": res_grid}}

#  增强型仿真系统
class CoevolutionSim:
    def __init__(self, config=None, size=50, model_name="llama3.1:8b"):
        self.size = size
        self.model_name = model_name
        self.api_url, self.host, self.port = get_ollama_url()

        # 导入配置
        if config:
            self.physics = config['physics']
            self.env_grids = config['grids']
            # 环境层：0:温度, 1:资源(甲烷/有机物), 2:关键代谢产物
            self.env = np.zeros((size, size, 3))
            self.env[:, :, 0] = self.env_grids['temp']
            self.env[:, :, 1] = self.env_grids['res']
            self.env[:, :, 2] = 0.01
        else:
            # 默认回退到 Titan 基础环境
            self.env = np.zeros((size, size, 3))
            self.env[:, :, 0] = -179.0  # 土卫六表面温度
            self.env[:, :, 1] = 1.0
            self.env[:, :, 2] = 0.05

        # 导入/初始化土卫六生物属性
        self.species = {
            "name": "乙烯基生命体 (Azotosome)",
            "metabolic_rate": 0.05,
            "temp_optimum": -179.0,
            "temp_tolerance": 20.0,
            "o2_impact": 0.0,  # 土卫六生物不产氧，设为0或改为化学影响
            "o2_tolerance": 999.0,  # 对氧气不敏感（或设为极高避免报错）
            "res_dependency": "Liquid Methane",
            "generation": 0,
            "history": ["基础甲烷代谢"]
        }

        # 必须初始化 biomass 数组（你之前的代码在 update 里用了 self.biomass 但 init 里没看到）
        self.biomass = np.zeros((size, size))
        mid = size // 2
        self.biomass[mid - 3:mid + 3, mid - 3:mid + 3] = 0.5

    def _call_ai_evolution(self):
        avg_temp = np.mean(self.env[:, :, 0])
        # 针对土卫六调整 Prompt，告诉 AI 这是一个非水基生命环境
        prompt = (
            f"环境报告：土卫六表面，平均温度{avg_temp:.1f}C。\n"
            f"当前生物：{self.species['name']}，依靠液态甲烷代谢。\n"
            f"请根据环境压力（如局部变暖或资源匮乏）进行协同进化，严格返回JSON："
            f"{{\"name\":\"新种名\",\"trait\":\"新特性\",\"temp_mod\":0.0,\"res_mod\":0.05,\"reason\":\"进化理由\"}}"
        )

        try:
            response = requests.post(f"{self.api_url}/chat/completions",
                                     json={
                                         "model": self.model_name,
                                         "messages": [{"role": "user", "content": prompt}],
                                         "temperature": 0.7
                                     }, timeout=5)
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            match = re.search(r'\{.*\}', content, re.DOTALL)
            return json.loads(match.group()) if match else None
        except:
            return None

    def update(self):
        # 1. 温度自然回归平衡 (Titan 表面约为 -179C)
        target_temp = self.physics.get('Surface_Temp', -179.0)
        self.env[:, :, 0] += (target_temp - self.env[:, :, 0]) * 0.01
        self.env[:, :, 0] += np.random.normal(0, 0.2, (self.size, self.size))

        # 2. 资源再生
        self.env[:, :, 1] = np.clip(self.env[:, :, 1] + 0.005, 0, 2.0)

        # 3. 计算适宜度
        temp_dist = np.abs(self.env[:, :, 0] - self.species['temp_optimum'])
        temp_fitness = np.exp(-(temp_dist ** 2) / (2 * self.species['temp_tolerance'] ** 2))

        # 修正 KeyError：如果物种没有 o2_tolerance，默认不产生压力
        o2_tol = self.species.get('o2_tolerance', 1.0)
        o2_stress = np.clip(self.env[:, :, 2] / o2_tol, 0, 2)
        fitness = temp_fitness * (1.0 / (1.0 + o2_stress))

        # 4. 代谢过程
        actual_consume = np.minimum(self.biomass * self.species['metabolic_rate'], self.env[:, :, 1])
        self.env[:, :, 1] -= actual_consume

        # 利基构造：生物产热
        # 在极度寒冷的 Titan，产热对生物是有益的（提高局部温度接近 optimum）
        self.env[:, :, 0] += actual_consume * 5.0
        # 假设代谢产生某种化学物质（占据原氧气通道）
        self.env[:, :, 2] += actual_consume * self.species.get('o2_impact', 0.01)

        # 5. 生长与扩散
        self.biomass = self.biomass + actual_consume * 2.0 - self.biomass * 0.1 * (1.2 - fitness)
        self.biomass = np.clip(self.biomass, 0, 10)

        # 扩散
        kernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
        self.biomass = convolve2d(self.biomass, kernel, mode='same')

    def run(self, steps=2000):  # 增加步数
        plt.ion()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        try:
            for i in range(steps):
                self.update()
                if i % 100 == 0:
                    print(f"当前进度: {i}/{steps}...")

                if i > 0 and i % 60 == 0:
                    evo = self._call_ai_evolution()
                    if evo:
                        self.species['name'] = evo.get('name', self.species['name'])
                        self.species['temp_optimum'] += evo.get('temp_mod', 0)
                        # 进化出耐氧性或更高的氧气产量
                        self.species['o2_tolerance'] += evo.get('o2_mod', 0.01)
                        self.species['history'].append(evo.get('trait', '适应性突变'))
                        self.species['generation'] += 1
                        print(f"[{i}] 事件: {evo.get('reason', '突变发生')}")

                #可视化
                titles = [f"生物量: {self.species['name']}", "环境温度 (代谢产热影响)", "氧气浓度 (利基构造)"]
                datas = [self.biomass, self.env[:, :, 0], self.env[:, :, 2]]
                cmaps = ['YlGn', 'hot', 'Blues']
                vmins = [0, 15, 0]
                vmaxs = [5, 45, 1]

                for idx, ax in enumerate(axes):
                    ax.clear()
                    im = ax.imshow(datas[idx], cmap=cmaps[idx], vmin=vmins[idx], vmax=vmaxs[idx])
                    ax.set_title(titles[idx])

                plt.pause(0.01)


        except KeyboardInterrupt:
            print("用户手动停止模拟")
        finally:
            plt.ioff()
            print("模拟完成，窗口将保持开启。")
            plt.show(block=True)  # 保证程序不退出


if __name__ == "__main__":
    # 初始化加载器
    loader = PlanetEnvironmentLoader(target_size=50)
    # 尝试加载 Titan 的 Excel 配置
    # 文件不存在，loader 里的 _generate_fallback_data 也会提供基础数据
    config = loader.load_planet_config("planet_data.xlsx")

    sim = CoevolutionSim(config=config, model_name="llama3.1:8b")
    sim.run(steps=2000)
    plt.show(block=True)
