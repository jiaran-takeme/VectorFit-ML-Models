import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib


class ConjugateClassifier(nn.Module):
    def __init__(self):
        super(ConjugateClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)


class PolePredictor(nn.Module):
    def __init__(self, output_dim):
        super(PolePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# 非共轭场景专用留数模型（9输入→3输出，LeakyReLU+交叉特征）
class NonConjResiduePredictor(nn.Module):
    def __init__(self, output_dim):
        super(NonConjResiduePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 256),  # 非共轭场景：9输入（3器件+3极点+3交叉）
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),  # 匹配训练代码的LeakyReLU
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, output_dim)  # 3输出（3个留数）
        )

    def forward(self, x):
        return self.model(x)


# 共轭场景原留数模型（3输入→3输出，ReLU，完全保留原逻辑）
class ConjResiduePredictor(nn.Module):
    def __init__(self, output_dim):
        super(ConjResiduePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 256),  # 共轭场景：保持原3输入（仅器件参数）
            nn.BatchNorm1d(256),
            nn.ReLU(),  # 保留原ReLU激活
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)  # 3输出（3个留数）
        )

    def forward(self, x):
        return self.model(x)


class PoleResiduePredictor:
    """三极点和留数预测器：非共轭场景用9输入留数模型，共轭保持原逻辑（完全按需求）"""

    def __init__(self,
                 # 分类器相关路径（按你设定保留）
                 classifier_model_path='best_conjugate_classifier.pth',
                 classifier_scaler_path='classifier_scaler.pkl',
                 # 共轭场景模型路径（按你设定保留，原逻辑）
                 conj_pole_model_path='./3pole_conjugate/best_conj_pole_predictor.pth',
                 conj_pole_info_path='./3pole_conjugate/best_conj_model_info.npy',
                 conj_res_model_path='./3res_conjugate/best_conj_residue_predictor_no_offset.pth',
                 conj_res_info_path='./3res_conjugate/best_conj_residue_info_no_offset.npy',
                 # 非共轭场景模型路径（按你设定保留，9输入模型）
                 nonconj_pole_model_path='./3pole_noconjugate/best_pole_predictor.pth',
                 nonconj_pole_info_path='./3pole_noconjugate/best_model_info.npy',
                 nonconj_res_model_path='./3res_noconjugate/best_residue_predictor_9feat_leakyrelu.pth',
                 nonconj_res_info_path='./3res_noconjugate/best_residue_info_9feat_leakyrelu.npy'):

        # -------------------------- 1. 加载分类模型（不变） --------------------------
        self.classifier = ConjugateClassifier()
        self.classifier.load_state_dict(torch.load(classifier_model_path, map_location='cpu', weights_only=True))
        self.classifier.eval()
        self.classifier_scaler = joblib.load(classifier_scaler_path)

        # -------------------------- 2. 加载共轭场景模型（完全保留原逻辑） --------------------------
        self.conj_pole_model = PolePredictor(output_dim=3)
        self.conj_pole_model.load_state_dict(torch.load(conj_pole_model_path, map_location='cpu', weights_only=True))
        self.conj_pole_model.eval()
        self.conj_pole_info = np.load(conj_pole_info_path, allow_pickle=True).item()

        # 共轭留数：用原3输入模型（不修改）
        self.conj_res_model = ConjResiduePredictor(output_dim=3)
        self.conj_res_model.load_state_dict(torch.load(conj_res_model_path, map_location='cpu', weights_only=True))
        self.conj_res_model.eval()
        self.conj_res_info = np.load(conj_res_info_path, allow_pickle=True).item()

        # -------------------------- 3. 加载非共轭场景模型（核心：9输入留数模型） --------------------------
        self.nonconj_pole_model = PolePredictor(output_dim=3)
        self.nonconj_pole_model.load_state_dict(
            torch.load(nonconj_pole_model_path, map_location='cpu', weights_only=True))
        self.nonconj_pole_model.eval()
        self.nonconj_pole_info = np.load(nonconj_pole_info_path, allow_pickle=True).item()

        # 非共轭留数：用9输入模型（按需求修改）
        self.nonconj_res_model = NonConjResiduePredictor(output_dim=3)
        self.nonconj_res_model.load_state_dict(
            torch.load(nonconj_res_model_path, map_location='cpu', weights_only=True))
        self.nonconj_res_model.eval()
        self.nonconj_res_info = np.load(nonconj_res_info_path, allow_pickle=True).item()

    def _init_scaler(self, scaler_dict):
        """初始化标准化器（通用）"""
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_dict['mean'])
        scaler.scale_ = np.array(scaler_dict['std'])
        return scaler

    def _preprocess_3feat(self, cs, rd, rs, scaler_dict):
        """3输入特征预处理（共轭留数、所有极点模型用）"""
        X = np.array([[cs, rd, rs]])
        scaler = self._init_scaler(scaler_dict)
        return torch.FloatTensor(scaler.transform(X))

    def _create_nonconj_res_feats(self, cs, rd, rs, poles):
        """为非共轭场景创建9输入留数特征（3器件+3极点+3交叉）"""
        # 提取非共轭极点实部（非共轭极点为实数，直接取real）
        pole1_real = poles[0].real
        pole2_real = poles[1].real
        pole3_real = poles[2].real

        # 计算交叉特征（与训练代码完全一致）
        cs_pole2 = cs * pole2_real  # Cs×Pole2
        pole2_pole3 = pole2_real / (pole3_real + 1e-6)  # Pole2/Pole3（避免除零）
        cs_pole3 = cs * pole3_real  # Cs×Pole3

        # 组合9个特征（6基础+3交叉）
        return np.array([[
            cs, rd, rs,
            pole1_real, pole2_real, pole3_real,
            cs_pole2, pole2_pole3, cs_pole3
        ]])

    def _is_conjugate(self, cs, rd, rs):
        """判断场景（不变）"""
        X_scaled = torch.FloatTensor(self.classifier_scaler.transform(np.array([[cs, rd, rs]])))
        with torch.no_grad():
            output = self.classifier(X_scaled)
            _, pred = torch.max(output, 1)
        return bool(pred.item())

    # -------------------------- 共轭场景后处理（原逻辑不变） --------------------------
    def _postprocess_conj_poles(self, pred):
        pred_np = pred.detach().numpy()[0]
        scaler_pole = self._init_scaler(self.conj_pole_info['scaler_y'])
        poles_unscaled = scaler_pole.inverse_transform([pred_np])[0]
        pole1_real = poles_unscaled[0] - self.conj_pole_info.get('output_offset', 0)
        pole1_imag = poles_unscaled[1]
        pole3_real = poles_unscaled[2] - self.conj_pole_info.get('output_offset', 0)
        return (pole1_real + pole1_imag * 1j,
                pole1_real - pole1_imag * 1j,
                pole3_real + 0j)

    def _postprocess_conj_residues(self, pred):
        pred_np = pred.detach().numpy()[0]
        scaler_res = self._init_scaler(self.conj_res_info['scaler_res'])
        res_unscaled = scaler_res.inverse_transform([pred_np])[0]
        res1_real = res_unscaled[0]
        res1_imag = res_unscaled[1]
        res3_real = res_unscaled[2]
        return (res1_real + res1_imag * 1j,
                res1_real - res1_imag * 1j,
                res3_real + 0j)

    # -------------------------- 非共轭场景后处理（适配9输入模型） --------------------------
    def _postprocess_nonconj_poles(self, pred):
        """非共轭极点后处理（原逻辑不变）"""
        scaler = self._init_scaler(self.nonconj_pole_info['scaler_y'])
        poles_unscaled = scaler.inverse_transform(pred.detach().numpy())[0]
        return tuple(p - self.nonconj_pole_info.get('output_offset', 0) + 0j for p in poles_unscaled)

    def _postprocess_nonconj_residues(self, pred):
        """非共轭留数后处理（适配9输入模型：无偏移，用scaler_y）"""
        pred_np = pred.detach().numpy()[0]
        scaler_res = self._init_scaler(self.nonconj_res_info['scaler_y'])  # 用非共轭留数的scaler_y
        res_unscaled = scaler_res.inverse_transform([pred_np])[0]
        return tuple(r + 0j for r in res_unscaled)  # 非共轭留数为实数

    def predict(self, cs, rd, rs):
        """
        预测入口：非共轭场景用9输入留数模型，共轭保持原逻辑
        参数: cs（电容）、rd（电阻Rd）、rs（电阻Rs）
        返回: poles（3个复数极点）、residues（3个复数留数）
        """
        has_conj = self._is_conjugate(cs, rd, rs)

        if has_conj:
            # -------------------------- 共轭场景：完全保留原逻辑 --------------------------
            # 1. 预测极点（3输入→3输出）
            X_scaled_pole = self._preprocess_3feat(cs, rd, rs, self.conj_pole_info['scaler_X'])
            with torch.no_grad():
                pole_pred = self.conj_pole_model(X_scaled_pole)
            poles = self._postprocess_conj_poles(pole_pred)

            # 2. 预测留数（3输入→3输出，原模型）
            X_scaled_res = self._preprocess_3feat(cs, rd, rs, self.conj_res_info['scaler_X'])
            with torch.no_grad():
                res_pred = self.conj_res_model(X_scaled_res)
            residues = self._postprocess_conj_residues(res_pred)

        else:
            # -------------------------- 非共轭场景：9输入留数模型（新逻辑） --------------------------
            # 1. 先预测极点（3输入→3输出，不变）
            X_scaled_pole = self._preprocess_3feat(cs, rd, rs, self.nonconj_pole_info['scaler_X'])
            with torch.no_grad():
                pole_pred = self.nonconj_pole_model(X_scaled_pole)
            poles = self._postprocess_nonconj_poles(pole_pred)

            # 2. 构建9输入留数特征（器件参数+极点+交叉特征）
            nonconj_res_feats = self._create_nonconj_res_feats(cs, rd, rs, poles)
            # 3. 标准化9输入特征（用非共轭留数的scaler_X）
            scaler_res_X = self._init_scaler(self.nonconj_res_info['scaler_X'])
            X_scaled_res = torch.FloatTensor(scaler_res_X.transform(nonconj_res_feats))

            # 4. 预测留数（9输入→3输出，新模型）
            with torch.no_grad():
                res_pred = self.nonconj_res_model(X_scaled_res)
            residues = self._postprocess_nonconj_residues(res_pred)

        return poles, residues


# 使用示例
if __name__ == "__main__":
    # 初始化预测器（路径按你设定，无需修改）
    predictor = PoleResiduePredictor()

    # 输入器件参数（注意：Cs单位若为fF，需确认训练时是否一致，此处按你示例值）
    Cs = 288.11  # 若训练时Cs单位为F，需改为80e-15；若为fF则无需修改
    Rd = 402.98
    Rs = 322.38

    # 预测极点和留数
    poles, residues = predictor.predict(Cs, Rd, Rs)

    # 输出结果（修正原代码的语法错误）
    print("=" * 60)
    print(f"预测场景：{'共轭极点场景' if predictor._is_conjugate(Cs, Rd, Rs) else '非共轭极点场景'}")
    print("=" * 60)
    print("预测极点:")
    for i, pole in enumerate(poles, 1):
        print(f"极点{i}: {pole:.6f}")
    print("\n预测留数:")
    for i, res in enumerate(residues, 1):
        print(f"留数{i}: {res:.6f}")  # 修正原代码的语法错误（缺少闭合引号）
    print("=" * 60)