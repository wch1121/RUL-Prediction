# RUL-Prediction
Code release for our IEEE Transactions on Transportation Electrification paper An Exponential Transformer for Learning Interpretable Temporal Information in Remaining Useful Life Prediction of lithium-ion Battery  
Chenhan Wang, Zhengyi Bao, Huipin Lin, Zhiwei He, Mingyu Gao
# Introduction
Accurately predicting the remaining useful life (RUL) of lithium-ion batteries is crucial for ensuring the safety and health maintenance of electric vehicles. Although conventional Transformers have demonstrated superior performance in various scenarios, they encounter significant challenges when applied to RUL estimation. Specifically, these models often lack the ability to decompose and interpret raw temporal data effectively, leading to underutilization of critical information. Additionally, the prolonged prediction times associated with these models can increase the risks associated with safe battery operation. To address these issues while preserving the strengths of Transformer-based temporal modeling, this paper introduces a novel Exponential Transformer designed to learn interpretable temporal features. Given the declining trend in battery capacity data and the presence of fluctuating information, this paper first extracts the level component and the fluctuation component from the original data, thereby decomposing the capacity information into interpretable time-series features. Furthermore, we propose an exponential attention mechanism to replace the self-attention mechanism, which enhances RUL prediction accuracy and efficiency by focusing on similar data features through exponential decay. Extensive experiments conducted on several aging datasets demonstrate that the proposed network achieves effective results in both single-step and multi-step prediction tasks. 
# Installation
Create environment and activate  
```
conda create --name etsformer python=3.8  
conda activate etsformer
```
Install requirements
```
pip install -r requirement.txt
```
# Training and Testing
```
python3.8 main.py
```
# citation
If you find ETSformer useful, please consider citing:
```javascript
@ARTICLE{wang2025exponential,
  author={Wang, Chenhan and Bao, Zhengyi and Lin, Huipin and He, Zhiwei and Gao, Mingyu},
  journal={IEEE Transactions on Transportation Electrification}, 
  title={An Exponential Transformer for Learning Interpretable Temporal Information in Remaining Useful Life Prediction of lithium-ion Battery}, 
  year={2025},
  volume={},
  number={},
  pages={1-12},
  keywords={Batteries,Transformers,Data models,Predictive models,Market research,Degradation,Data mining,Fluctuations,Attention mechanisms,Feature extraction},
  doi={10.1109/TTE.2025.3534146}}
}
```
