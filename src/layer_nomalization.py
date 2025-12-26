import numpy as np


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """
    Layer NormalizationをNumPyで実装

    Parameters:
    x: 入力データ (shape: [batch_size, features] または [batch_size, seq_len, features])
    gamma: スケーリングパラメータ (Noneの場合は1.0で初期化)
    beta: シフトパラメータ (Noneの場合は0.0で初期化)
    eps: ゼロ除算を避けるための微小値
    """
    # 1. 最後の次元（特徴量次元）に沿って平均を計算
    # keepdims=Trueにすることで、元のxと計算ができる形状を維持します
    mean = np.mean(x, axis=-1, keepdims=True)

    # 2. 最後の次元に沿って分散を計算
    var = np.var(x, axis=-1, keepdims=True)

    # 3. 正規化 (平均0, 分散1に変換)
    x_normalized = (x - mean) / np.sqrt(var + eps)

    # 4. gammaとbetaの初期化（指定がない場合）
    if gamma is None:
        gamma = np.ones(x.shape[-1])
    if beta is None:
        beta = np.zeros(x.shape[-1])

    # 5. 再スケーリングとシフト
    out = gamma * x_normalized + beta

    return out


# --- 動作確認 ---
# 形状: (バッチサイズ2, 特徴量3) の適当な入力
x_in = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 40.0]])

output = layer_norm(x_in)

print("入力データ:\n", x_in)
print("\nLayer Norm後の出力:\n", output)

# 各サンプルの平均がほぼ0、分散がほぼ1になっているか確認
print("\n出力の平均（行ごと）:", np.mean(output, axis=-1))
print("出力の分散（行ごと）:", np.var(output, axis=-1))
