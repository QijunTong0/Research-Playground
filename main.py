"""
1次元ランジェバン方程式の時空間 (t-x) シミュレーション
背景: 時空平面全体におけるドリフトベクトル場（決定論的な流れ）
前景: その場の中を進む確率的なサンプルパス（一本ずつ順次描画）
"""

import numpy as np

# 明示的なインポート
from manim import (
    DOWN,
    GRAY,
    # 定数・色・配置
    RED,
    TEAL,
    UL,
    UP,
    YELLOW,
    Arrow,
    Axes,
    Dot,
    FadeIn,
    Scene,
    Text,
    TracedPath,
    ValueTracker,
    VGroup,
    # アニメーション用クラス
    Write,
    config,
    linear,
)


class LangevinSpaceTime(Scene):
    def construct(self):
        # --- 1. パラメータ設定 ---
        np.random.seed(42)
        n_particles = 8  # サンプルパスの本数
        t_max = 10.0  # シミュレーション終了時間
        dt = 0.025  # 時間刻み幅 (より細かく変更)
        sigma = 0.6  # ノイズの強さ

        # 凸ポテンシャル（調和振動子）の中心を動かすパラメータ
        # U(x, t) = 0.5 * k * (x - center(t))^2
        # F(x, t) = -k * (x - center(t))
        k_spring = 2.0  # バネ定数（引き戻す力）
        amp = 1.5  # 中心の振幅
        omega = 1.5  # 中心の振動数

        # --- 2. 物理関数の定義 ---
        def get_center(t: float) -> float:
            """ポテンシャルの中心（谷底）の位置"""
            return amp * np.sin(omega * t)

        def force(x: float, t: float) -> float:
            """
            時刻 t, 位置 x における復元力
            F = -dU/dx = -k * (x - x_center)
            """
            return -k_spring * (x - get_center(t))

        # --- 3. 描画オブジェクト: 座標軸 (t-x 平面) ---
        axes = Axes(
            x_range=[0, t_max, 1],
            y_range=[-3, 3, 1],
            x_length=11,
            y_length=6,
            axis_config={
                "include_tip": True,
                "color": GRAY,
            },
        ).add_coordinates()

        labels = axes.get_axis_labels(x_label="t", y_label="x")

        # --- 4. 背景ベクトル場の描画 ---
        vector_field = VGroup()

        # グリッド生成
        t_steps = np.linspace(0.5, t_max - 0.5, 14)
        x_steps = np.linspace(-2.5, 2.5, 12)

        for t_val in t_steps:
            for x_val in x_steps:
                f = force(x_val, t_val)

                # 時空上のベクトル: (dt, dx) 方向
                # 視覚的バランスのためのスケーリング
                vec_x = f * 0.3

                start_point = axes.c2p(t_val, x_val)
                end_point = axes.c2p(t_val, x_val + vec_x)

                # 中心に向かう力は青、外へ向かう（今回は凸なので基本ないが）力は赤
                # ここでは力の向きではなく、ポテンシャルの谷（中心）との距離で色付けしてみる
                # あるいは単純に力の向きで色付け
                color = TEAL if f < 0 else RED  # 下向き(負)なら青、上向き(正)なら赤

                # 矢印作成
                arrow = Arrow(
                    start=start_point,
                    end=end_point,
                    buff=0,
                    max_tip_length_to_length_ratio=0.25,
                    stroke_width=2,
                    color=color,
                    stroke_opacity=0.6,
                )
                vector_field.add(arrow)

        # タイトル
        title = Text("Langevin Dynamics (Convex Potential)", font_size=32).to_edge(UP)
        subtitle = Text(
            "One-by-one sample paths in Space-Time", font_size=20, color=GRAY
        ).next_to(title, DOWN)

        # 背景のセットアップ
        self.add(axes, labels)
        self.play(FadeIn(vector_field), Write(title), Write(subtitle), run_time=1.5)

        # --- 5. サンプルパスのシミュレーション (一本ずつ) ---

        # 過去のパスを保存しておくグループ
        finished_paths = VGroup()
        self.add(finished_paths)

        for i in range(n_particles):
            # 進捗表示
            prog_text = Text(f"Path: {i + 1}/{n_particles}", font_size=24, color=YELLOW)
            prog_text.to_corner(UL)
            self.add(prog_text)

            # シミュレーション用タイマー
            t_tracker = ValueTracker(0.0)

            # 粒子の初期化 (t=0, xはランダムまたは0)
            # 凸関数なので、初期値がばらついていても中心に集まっていく様子が見えるはず
            start_x = np.random.uniform(-2.5, 2.5)

            dot = Dot(color=YELLOW, radius=0.08)
            dot.move_to(axes.c2p(0, start_x))

            # 軌跡
            trace = TracedPath(
                dot.get_center,
                stroke_color=YELLOW,
                stroke_opacity=0.8,  # 少し濃く
                stroke_width=2.5,
                dissipating_time=None,
            )

            self.add(trace, dot)

            # 粒子の状態データ [現在のシミュレーション時刻t, 現在の位置x]
            # Updater内で参照・更新するためにオブジェクトの属性として持たせる
            dot.sim_t = 0.0
            dot.sim_x = start_x

            # Updater関数
            def update_particle(mob):
                target_t = t_tracker.get_value()
                current_t = mob.sim_t
                current_x = mob.sim_x

                # 前回の時刻からターゲット時刻まで dt 刻みで積分を進める
                while current_t < target_t:
                    # 時間幅の調整（最後のステップがdt未満の場合の処理）
                    step_dt = min(dt, target_t - current_t)
                    if step_dt <= 1e-6:
                        break

                    # Euler-Maruyama法
                    f = force(current_x, current_t)
                    noise = np.random.normal(0, 1)

                    dx = f * step_dt + sigma * np.sqrt(step_dt) * noise

                    current_x += dx
                    current_t += step_dt

                # 状態更新
                mob.sim_t = current_t
                mob.sim_x = current_x

                # 描画位置更新
                mob.move_to(axes.c2p(current_t, current_x))

            dot.add_updater(update_particle)

            # アニメーション実行 (t=0 -> t_max)
            # run_timeを短めにしてテンポよく
            self.play(
                t_tracker.animate.set_value(t_max), run_time=3.0, rate_func=linear
            )

            # 完了処理
            dot.remove_updater(update_particle)

            # Dotは消すが、Traceは残したい。
            # Traceを静的なPath（VMobject）に変換するか、
            # Dotを透明にして残すか。ここではDotを透明にして残す（Traceの参照を維持するため）
            dot.set_opacity(0)

            # 今回のTraceをグループに追加（管理用）
            finished_paths.add(trace)

            # 進捗テキスト削除
            self.remove(prog_text)

        self.wait(1)


if __name__ == "__main__":
    config.quality = "medium_quality"
    scene = LangevinSpaceTime()
    scene.render()
