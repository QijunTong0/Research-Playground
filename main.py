"""
1次元ランジェバン方程式の時空間 (t-x) シミュレーション
テーマ: 経路の分岐と、それに伴う確率分布の時間発展
"""

import numpy as np

# 明示的なインポート
from manim import (
    BLUE,
    GRAY,
    # 定数・色・配置
    RED,
    TEAL,
    UL,
    UP,
    UR,
    YELLOW,
    Arrow,
    Axes,
    # アニメーション用クラス
    Create,
    DashedVMobject,  # 点線用
    DecimalNumber,
    Dot,
    FadeIn,
    MathTex,
    Scene,
    Text,
    TracedPath,
    ValueTracker,
    VGroup,
    VMobject,  # 修正: VMobjectのインポート漏れを追加
    Write,
    linear,
)


# --- 共通の物理パラメータ・関数 ---
def get_bifurcation_force(x, t, t_bifurcation=3.0):
    """分岐の力場計算"""
    # 分岐パラメータ a(t)
    a = 0.8 * (t - t_bifurcation)
    # 力 F(x) = -x^3 + a(t)x
    return 0.8 * (-(x**3) + a * x)


def compute_kde_distribution(particle_positions, bandwidth=0.3, num_points=80):
    """カーネル密度推定(KDE)による分布計算"""
    x_grid = np.linspace(-3, 3, num_points)
    density = np.zeros_like(x_grid)
    n = len(particle_positions)

    for x_p in particle_positions:
        density += np.exp(-0.5 * ((x_grid - x_p) / bandwidth) ** 2)

    # 正規化とスケーリング (高さ調整)
    density = density / (n * bandwidth * np.sqrt(2 * np.pi)) * 3.0
    return x_grid, density


class LangevinSpaceTime(Scene):
    """
    シーン1: 時空図上でのサンプルパスの分岐（一本ずつ描画）
    """

    def construct(self):
        # --- パラメータ ---
        np.random.seed(42)
        n_particles = 10
        t_max = 10.0
        dt = 0.02
        sigma = 0.5
        t_bifurcation = 3.0

        # --- 描画オブジェクト ---
        axes = Axes(
            x_range=[0, t_max, 1],
            y_range=[-3, 3, 1],
            x_length=11,
            y_length=6,
            axis_config={"include_tip": True, "color": GRAY},
        ).add_coordinates()
        labels = axes.get_axis_labels(x_label="t", y_label="x")

        # 背景ベクトル場
        vector_field = VGroup()
        t_steps = np.linspace(0.5, t_max - 0.5, 18)
        x_steps = np.linspace(-2.5, 2.5, 15)
        for t_val in t_steps:
            for x_val in x_steps:
                f = get_bifurcation_force(x_val, t_val, t_bifurcation)
                vec_t = 0.3
                vec_x = f * 0.15
                color = RED if f > 0 else TEAL
                arrow = Arrow(
                    start=axes.c2p(t_val, x_val),
                    end=axes.c2p(t_val + vec_t, x_val + vec_x),
                    buff=0,
                    max_tip_length_to_length_ratio=0.35,
                    max_stroke_width_to_length_ratio=10,
                    stroke_width=1.5,
                    color=color,
                    stroke_opacity=0.4,
                )
                vector_field.add(arrow)

        # 修正: タイトル削除
        # タイトルやグラフをフェードインで表示
        self.play(FadeIn(axes), Write(labels), run_time=1.5)
        self.play(FadeIn(vector_field), run_time=1.0)

        # --- シミュレーション ---
        finished_paths = VGroup()
        self.add(finished_paths)

        for i in range(n_particles):
            prog_text = Text(
                f"試行: {i + 1}/{n_particles}", font_size=24, color=YELLOW
            ).to_corner(UL)
            self.add(prog_text)

            t_tracker = ValueTracker(0.0)
            start_x = np.random.normal(0, 0.2)

            dot = Dot(color=YELLOW, radius=0.08)
            dot.move_to(axes.c2p(0, start_x))

            trace = TracedPath(
                dot.get_center,
                stroke_color=YELLOW,
                stroke_opacity=0.8,
                stroke_width=2.0,
                dissipating_time=None,
            )
            self.add(trace, dot)

            dot.sim_t = 0.0
            dot.sim_x = start_x

            def update_particle(mob):
                target_t = t_tracker.get_value()
                current_t = mob.sim_t
                current_x = mob.sim_x

                while current_t < target_t:
                    step_dt = min(dt, target_t - current_t)
                    if step_dt <= 1e-6:
                        break

                    f = get_bifurcation_force(current_x, current_t, t_bifurcation)
                    noise = np.random.normal(0, 1)
                    dx = f * step_dt + sigma * np.sqrt(step_dt) * noise

                    current_x += dx
                    current_t += step_dt

                    if current_x > 3.5:
                        current_x = 3.5
                    if current_x < -3.5:
                        current_x = -3.5

                mob.sim_t = current_t
                mob.sim_x = current_x
                mob.move_to(axes.c2p(current_t, current_x))

            dot.add_updater(update_particle)
            self.play(
                t_tracker.animate.set_value(t_max), run_time=1.5, rate_func=linear
            )

            dot.remove_updater(update_particle)
            dot.set_opacity(0)
            finished_paths.add(trace)
            self.remove(prog_text)

        self.wait(1)


class LangevinDistribution(Scene):
    """
    シーン2: 多数の粒子による分布の時間発展
    時空図 (t-x) 上に、その時刻における分布曲線を重ねて表示する
    """

    def construct(self):
        # --- パラメータ ---
        np.random.seed(999)
        n_particles = 150  # 粒子数
        t_max = 10.0
        dt = 0.05  # 描画更新用ステップ
        sim_dt = 0.01  # 物理計算用サブステップ
        sigma = 0.5
        t_bifurcation = 3.0

        # --- 描画オブジェクト ---
        # 単一の時空図
        axes = Axes(
            x_range=[0, t_max, 1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": True, "color": GRAY},
        ).add_coordinates()

        labels = axes.get_axis_labels(x_label="t", y_label="x")

        # 修正: タイトル削除
        # アニメーションで表示
        self.play(FadeIn(axes), Write(labels), run_time=1.5)

        # --- 粒子の初期化 ---
        particles = VGroup()
        particle_data = []  # [current_t, current_x]

        for _ in range(n_particles):
            # 初期位置のバラつき
            start_x = np.random.normal(0, 0.3)

            dot = Dot(color=YELLOW, radius=0.04, fill_opacity=0.5)
            dot.move_to(axes.c2p(0, start_x))

            # 軌跡（薄く残す）
            trace = TracedPath(
                dot.get_center,
                stroke_color=YELLOW,
                stroke_opacity=0.2,  # 薄くして分布曲線を見やすくする
                stroke_width=1.0,
                dissipating_time=None,
            )

            particles.add(dot)
            self.add(trace)  # traceは個別にadd
            # dotはparticlesごとaddするため、個別addは不要

            particle_data.append(start_x)

        # 粒子は最初は見えない状態からFadeInさせる
        self.play(FadeIn(particles), run_time=1.0)

        # --- 分布曲線の初期化 ---
        # axes 上に描画する線。
        # 時刻 t の位置に、xに応じた密度分布を「右向きの山」として描画する
        distribution_curve = axes.plot_line_graph(
            x_values=[0, 0],
            y_values=[-3, 3],
            line_color=BLUE,
            add_vertex_dots=False,
            stroke_width=4,
        )

        # MathTexを使用してLaTeXレンダリング
        dist_label = MathTex(r"P(x, t)", font_size=24, color=BLUE).next_to(
            distribution_curve, UP
        )

        # 分布曲線とラベルもアニメーションで表示
        self.play(Create(distribution_curve), Write(dist_label), run_time=1.0)

        # --- シミュレーション用トラッカー ---
        t_tracker = ValueTracker(0.0)
        # 修正: タイトルがないので、位置を右上に固定
        time_label = DecimalNumber(
            0, num_decimal_places=1, include_sign=False
        ).to_corner(UR)
        self.add(time_label)

        # --- アップデータ ---
        def update_scene(mob):
            # 現在のアニメーション時刻
            current_t = t_tracker.get_value()
            time_label.set_value(current_t)

            # 1. 粒子の物理計算と移動
            steps = int(dt / sim_dt)
            for _ in range(steps):
                if current_t >= t_max:
                    break

                x_vals = np.array(particle_data)

                # 力の計算
                f = get_bifurcation_force(x_vals, current_t, t_bifurcation)
                noise = np.random.normal(0, 1, n_particles)

                dx = f * sim_dt + sigma * np.sqrt(sim_dt) * noise
                x_vals += dx

                # 境界条件
                x_vals = np.clip(x_vals, -3.5, 3.5)

                # データ更新
                for i in range(n_particles):
                    particle_data[i] = x_vals[i]

            # ドットの描画位置更新
            for i, dot in enumerate(particles):
                dot.move_to(axes.c2p(current_t, particle_data[i]))

            # 2. 分布の計算と描画 (同じグラフ上に重ねる)
            # 共通化したKDE関数を使用
            x_grid, density = compute_kde_distribution(particle_data)

            # スケーリング
            scale_factor = 1.0
            t_coords = current_t + density * scale_factor

            new_curve = axes.plot_line_graph(
                x_values=t_coords,
                y_values=x_grid,
                line_color=BLUE,
                add_vertex_dots=False,
                stroke_width=4,
            )
            distribution_curve.become(new_curve)

            # ラベルも追従させる
            dist_label.next_to(axes.c2p(current_t, 3.2), UP)

        particles.add_updater(update_scene)

        self.play(t_tracker.animate.set_value(t_max), run_time=8.0, rate_func=linear)

        particles.remove_updater(update_scene)
        self.wait(1)


class LangevinInverseProblem(Scene):
    """
    シーン3: 逆問題デモンストレーション
    1. 未来の周辺分布(t=2,4,6,8)を「観測データ」として事前に描画
    2. t=0からシミュレーションを行い、分布が観測データに一致していく様子を見せる
    """

    def construct(self):
        # --- パラメータ (Scene2と同じにして整合性を取る) ---
        seed = 999
        n_particles = 150
        t_max = 10.0
        dt = 0.05
        sim_dt = 0.01
        sigma = 0.5
        t_bifurcation = 3.0

        target_times = [2.0, 4.0, 6.0, 8.0]  # 分布を表示する時刻

        # --- 描画オブジェクト ---
        axes = Axes(
            x_range=[0, t_max, 1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": True, "color": GRAY},
        ).add_coordinates()
        labels = axes.get_axis_labels(x_label="t", y_label="x")

        # 修正: タイトルとサブタイトルを削除
        self.play(FadeIn(axes), Write(labels), run_time=1.5)

        # --- 1. 事前計算 (Target Distributions) ---
        # シミュレーションをバックグラウンドで回して、ターゲット時刻の分布を計算する
        np.random.seed(seed)  # 同じシードを使用

        # 粒子初期化（計算用）
        sim_particles = np.random.normal(0, 0.3, n_particles)

        target_curves = VGroup()

        current_sim_t = 0.0
        target_idx = 0

        # ステップごとのループ（描画はしない）
        while current_sim_t < t_max and target_idx < len(target_times):
            # 次のターゲット時刻までのステップ数
            next_target_t = target_times[target_idx]

            # 到達していないなら進める
            while current_sim_t < next_target_t:
                f = get_bifurcation_force(sim_particles, current_sim_t, t_bifurcation)
                noise = np.random.normal(0, 1, n_particles)
                dx = f * sim_dt + sigma * np.sqrt(sim_dt) * noise
                sim_particles += dx
                sim_particles = np.clip(sim_particles, -3.5, 3.5)
                current_sim_t += sim_dt

            # ターゲット時刻に到達したので分布を作成
            x_grid, density = compute_kde_distribution(sim_particles)

            # 分布曲線の生成 (点線、薄い色)
            scale_factor = 1.0
            t_coords = next_target_t + density * scale_factor

            # VMobjectを作成
            points = [axes.c2p(t, x) for t, x in zip(t_coords, x_grid)]

            # 1本の連続した線として作成
            curve_vm = VMobject()
            curve_vm.set_points_as_corners(points)
            curve_vm.set_color(GRAY)

            # DashedVMobjectで点線化
            dashed_curve = DashedVMobject(curve_vm, num_dashes=25, dashed_ratio=0.5)

            # ラベル
            label = MathTex(
                f"t={int(next_target_t)}", font_size=20, color=GRAY
            ).next_to(axes.c2p(next_target_t, 3.2), UP)

            target_curves.add(dashed_curve, label)
            target_idx += 1

        # ターゲット分布を表示
        self.play(Create(target_curves), run_time=2.0)
        self.wait(0.5)

        # --- 2. リアルタイムシミュレーション (Animation) ---
        # シードをリセットして、同じ挙動を再現
        np.random.seed(seed)

        particles = VGroup()
        particle_data = []

        # 粒子の初期化 (描画用)
        for _ in range(n_particles):
            start_x = np.random.normal(0, 0.3)
            dot = Dot(color=YELLOW, radius=0.04, fill_opacity=0.5)
            dot.move_to(axes.c2p(0, start_x))

            trace = TracedPath(
                dot.get_center,
                stroke_color=YELLOW,
                stroke_opacity=0.2,
                stroke_width=1.0,
                dissipating_time=None,
            )
            particles.add(dot)
            self.add(trace)
            particle_data.append(start_x)

        self.play(FadeIn(particles), run_time=1.0)

        # リアルタイム分布曲線
        realtime_curve = axes.plot_line_graph(
            x_values=[0, 0],
            y_values=[-3, 3],
            line_color=BLUE,
            add_vertex_dots=False,
            stroke_width=4,
        )
        dist_label = MathTex(r"P_{sim}(x, t)", font_size=24, color=BLUE).next_to(
            realtime_curve, UP
        )

        self.play(Create(realtime_curve), Write(dist_label), run_time=1.0)

        # トラッカー
        t_tracker = ValueTracker(0.0)
        # 修正: タイトルがないので、位置を右上に固定
        time_label = DecimalNumber(
            0, num_decimal_places=1, include_sign=False
        ).to_corner(UR)
        self.add(time_label)

        def update_scene(mob):
            current_t = t_tracker.get_value()
            time_label.set_value(current_t)

            # 物理計算
            steps = int(dt / sim_dt)
            for _ in range(steps):
                if current_t >= t_max:
                    break
                x_vals = np.array(particle_data)
                f = get_bifurcation_force(x_vals, current_t, t_bifurcation)
                noise = np.random.normal(0, 1, n_particles)
                dx = f * sim_dt + sigma * np.sqrt(sim_dt) * noise
                x_vals += dx
                x_vals = np.clip(x_vals, -3.5, 3.5)
                for i in range(n_particles):
                    particle_data[i] = x_vals[i]

            for i, dot in enumerate(particles):
                dot.move_to(axes.c2p(current_t, particle_data[i]))

            # 分布更新
            x_grid, density = compute_kde_distribution(particle_data)
            scale_factor = 1.0
            t_coords = current_t + density * scale_factor

            new_curve = axes.plot_line_graph(
                x_values=t_coords,
                y_values=x_grid,
                line_color=BLUE,
                add_vertex_dots=False,
                stroke_width=4,
            )
            realtime_curve.become(new_curve)
            dist_label.next_to(axes.c2p(current_t, 3.2), UP)

        particles.add_updater(update_scene)

        self.play(t_tracker.animate.set_value(t_max), run_time=8.0, rate_func=linear)

        particles.remove_updater(update_scene)
        self.wait(1)
