import pandas as pd
import numpy as np

def generate_excel_template(filename="planet_data.xlsx"):
    with pd.ExcelWriter(filename) as writer:
        # 1. Physical 特征
        phys_df = pd.DataFrame({
            "Property": ["Name", "Mass", "Gravity", "Radius", "Albedo", "Axial_Tilt", "Distance"],
            "Value": ["Ares-Prime", 1.2, 11.2, 7000, 0.3, 23.5, 1.1]
        }).set_index("Property")
        phys_df.to_excel(writer, sheet_name='Physical')

        # 2. Atmosphere 成分
        atmos_df = pd.DataFrame({
            "Component": ["Nitrogen", "Oxygen", "CO2", "Pressure"],
            "Ratio": [0.70, 0.01, 0.25, 1.5]
        }).set_index("Component")
        atmos_df.to_excel(writer, sheet_name='Atmosphere')

        # 3. Temperature 网格 (10x10 初始矩阵)
        # 生成一个两极冷赤道热的简单模拟
        temp_data = np.zeros((10, 10))
        for i in range(10):
            temp_val = 30 - abs(i - 5) * 8  # 模拟纬度温差
            temp_data[i, :] = temp_val + np.random.normal(0, 2, 10)
        pd.DataFrame(temp_data).to_excel(writer, sheet_name='Temperature', header=False, index=False)

        # 4. Resources 网格
        res_data = np.random.uniform(0.5, 1.5, (10, 10))
        pd.DataFrame(res_data).to_excel(writer, sheet_name='Resources', header=False, index=False)

    print(f"模板已生成: {filename}")

if __name__ == "__main__":
    generate_excel_template()