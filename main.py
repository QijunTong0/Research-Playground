import numpy as np
from manim import *


class DynamicWaveLangevin(Scene):
    def construct(self):
        # --- 1. パラメータ設定 ---
        np.random.seed(999)
        n_particles = 5  # サンプルパスの本数（1本ずつ表示）
        duration = 8  # 1回のシミュレーション時間
        sigma = 0.8  # ノイズの強さ
        dt = 1 / 60  # タイムステップ

        # 波のパラメータ
        # U(x, t) = cos(k*x + w*t) + c*x^2
        k = 3.0  # 空間周波数（山谷の密度）
        w = 1.5  # 波の移動速度
        c = 0.1  # 閉じ込め項の係数

        t_tracker = ValueTracker(0)

        # --- 2. 関数定義 ---
        def potential(x, t):
            # 移動する波 + 緩やかな放物線
            return np.cos(k * x + w * t) + c * x**2

        def force(x, t):
            # F = -dU/dx
            # dU/dx = -k * sin(k*x + w*t) + 2*c*x
            # F = k * sin(k*x + w*t) - 2*c*x
            return k * np.sin(k * x + w * t) - 2 * c * x

        # --- 3. 描画オブジェクト ---
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 3, 1],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": False},
        ).add_coordinates()

        labels = axes.get_axis_labels(x_label="x", y_label="U(x, t)")

        # 時間表示
        time_label = DecimalNumber(
            0, num_decimal_places=2, include_sign=False, unit=" s"
        )
        time_label.to_corner(UR)
        time_text = Text("Time: ").next_to(time_label, LEFT)

        # ポテンシャルグラフ（初期化）
        pot_graph = axes.plot(lambda x: potential(x, 0), color=BLUE, x_range=[-3, 3])

        # ベクトル場（矢印）生成関数
        def get_arrows(t):
            group = VGroup()
            # 矢印の密度を少し上げる
            for x in np.linspace(-2.8, 2.8, 25):
                f = force(x, t)

                # 力が小さいときは描画しない（視認性向上）
                if abs(f) < 0.15:
                    continue

                # 矢印の長さと色
                # 力が強いほど長くするが、長すぎないようにclipする
                magnitude = np.clip(abs(f) * 0.15, 0.1, 0.5)
                color = RED if f > 0 else TEAL

                # 矢印の配置：X軸より少し下
                arrow_y = -1.5

                arrow = Arrow(
                    start=axes.c2p(x, arrow_y),
                    end=axes.c2p(x + (magnitude if f > 0 else -magnitude), arrow_y),
                    buff=0,
                    max_tip_length_to_length_ratio=0.4,
                    stroke_width=3,
                    color=color,
                )
                group.add(arrow)
            return group

        vector_field = get_arrows(0)
        vector_text = Text("Vector Field (Force)", font_size=24).next_to(
            vector_field, DOWN, buff=0.2
        )

        self.add(
            axes, labels, pot_graph, vector_field, vector_text, time_label, time_text
        )

        # --- 4. 環境の更新 (Updater) ---
        def update_env(mob):
            t = t_tracker.get_value()
            time_label.set_value(t)

            # ポテンシャル形状の更新
            mob.become(
                axes.plot(lambda x: potential(x, t), color=BLUE, x_range=[-3, 3])
            )

            # 矢印の更新
            new_arrows = get_arrows(t)
            vector_field.become(new_arrows)

        pot_graph.add_updater(update_env)

        # --- 5. サンプルパスのシミュレーション ---
        for i in range(n_particles):
            # リセット
            t_tracker.set_value(0)

            # カウンター表示
            run_text = Text(
                f"Path: {i + 1}/{n_particles}", font_size=30, color=YELLOW
            ).to_corner(UL)
            self.add(run_text)

            # 粒子の初期化
            # 初期位置をランダムに分散させる
            start_x = np.random.uniform(-1.0, 1.0)
            dot = Dot(color=YELLOW, radius=0.12)
            dot.move_to(axes.c2p(start_x, potential(start_x, 0)))
            # 軌跡を残すためのTracing
            trace = TracedPath(
                dot.get_center,
                stroke_color=YELLOW,
                stroke_opacity=0.5,
                stroke_width=2,
                dissipating_time=0.5,
            )
            self.add(trace, dot)

            dot.x_val = start_x

            # 粒子の動き更新
            def update_dot(mob, dt):
                t = t_tracker.get_value()
                x = mob.x_val

                # ランジェバン方程式
                f = force(x, t)
                noise = np.random.normal(0, 1)
                dx = f * dt + sigma * np.sqrt(dt) * noise

                new_x = x + dx

                # 画面端の処理（反射）
                if new_x > 3.0:
                    new_x = 3.0
                if new_x < -3.0:
                    new_x = -3.0

                mob.x_val = new_x
                # ポテンシャルの上に乗せる
                mob.move_to(axes.c2p(new_x, potential(new_x, t)))

            dot.add_updater(update_dot)

            # アニメーション再生
            self.play(
                t_tracker.animate.set_value(duration),
                run_time=duration,
                rate_func=linear,
            )

            # 後処理
            dot.remove_updater(update_dot)
            self.remove(dot, trace, run_text)
            self.wait(0.2)

        pot_graph.remove_updater(update_env)
