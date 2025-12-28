import manim
import numpy as np


class OfutonGravitySimulation(manim.ThreeDScene):
    def construct(self):
        # 1. タイトルの表示 (日本語のみ)
        title = manim.Text("布団重力場の動的解析", font_size=40).to_edge(manim.UP)
        subtitle = manim.Text(
            "一般相対論的 布団拘束シミュレーション", font_size=20, color=manim.BLUE
        ).next_to(title, manim.DOWN)
        self.add(title, subtitle)
        self.wait(1)
        self.play(manim.FadeOut(title), manim.FadeOut(subtitle))

        # 2. 3D座標系の設定
        axes = manim.ThreeDAxes()

        # 布団による時空の歪みを定義する関数
        # 深さを元の3/4に変更 (-3.0 -> -2.25)
        DEPTH_FACTOR = 2.25

        def gravity_function(u, v):
            r = np.sqrt(u**2 + v**2)
            z = -DEPTH_FACTOR * np.exp(-0.5 * r**2)
            return np.array([u, v, z])

        # 時空グリッド
        grid = manim.Surface(
            gravity_function,
            u_range=[-4, 4],
            v_range=[-4, 4],
            resolution=(20, 20),
            fill_opacity=0.2,
            checkerboard_colors=[manim.BLUE_E, manim.BLUE_D],
        )

        # --- 布団オブジェクトの構築 ---
        # 敷き布団 (Base mattress)
        mattress = (
            manim.Cube(side_length=1).scale([1.2, 1.8, 0.1]).set_color(manim.WHITE)
        )
        # 掛け布団 (Quilt/Kakebuton) - 色をオレンジに変更
        quilt = (
            manim.Cube(side_length=1)
            .scale([1.25, 1.4, 0.2])
            .set_color(manim.ORANGE)
            .move_to([0, -0.2, 0.15])
        )
        # 枕 (Pillow) - 色を灰色に変更
        pillow = (
            manim.Cube(side_length=1)
            .scale([0.6, 0.3, 0.1])
            .set_color(manim.GRAY)
            .move_to([0, 0.6, 0.1])
        )

        ofuton_group = manim.Group(mattress, quilt, pillow)

        # グリッドの底に合わせて配置位置を調整
        ofuton_group.move_to([0, 0, -2.05])

        # フォントサイズを大きくし、日本語に変更
        ofuton_label = manim.Text("巨大布団", font_size=42).next_to(
            ofuton_group, manim.DOWN, buff=0.5
        )

        # 3. カメラアングルの変更
        self.set_camera_orientation(phi=75 * manim.DEGREES, theta=-45 * manim.DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1)

        self.play(manim.Create(grid))
        self.play(manim.FadeIn(ofuton_group), manim.Write(ofuton_label))
        self.wait(1)

        # 4. 人間（粒子）の挙動
        # 棒人間 (Stick Figure) の構築
        head = manim.Sphere(radius=0.06, color=manim.YELLOW).move_to([0, 0, 0.08])  # 頭
        body = manim.Line(
            start=[0, 0, 0.08], end=[0, 0, -0.08], color=manim.YELLOW, stroke_width=4
        )  # 胴体
        arms = manim.Line(
            start=[-0.08, 0, 0.02],
            end=[0.08, 0, 0.02],
            color=manim.YELLOW,
            stroke_width=4,
        )  # 腕
        leg_l = manim.Line(
            start=[0, 0, -0.08],
            end=[-0.05, 0, -0.18],
            color=manim.YELLOW,
            stroke_width=4,
        )  # 左足
        leg_r = manim.Line(
            start=[0, 0, -0.08],
            end=[0.05, 0, -0.18],
            color=manim.YELLOW,
            stroke_width=4,
        )  # 右足

        human = manim.VGroup(head, body, arms, leg_l, leg_r)
        # 寝ている姿勢（水平）になるよう回転
        human.rotate(manim.PI / 2, axis=manim.RIGHT)

        # 軌跡の設定変更: dissipating_timeを長くして軌跡が残るようにする
        human_trail = manim.TracedPath(
            human.get_center,
            dissipating_time=20.0,
            stroke_opacity=[0, 1],
            stroke_width=3,
            stroke_color=manim.YELLOW,
        )
        self.add(human_trail)

        def update_human(mob, dt):
            if not hasattr(self, "internal_time"):
                self.internal_time = 0
            self.internal_time += dt

            t = self.internal_time
            decay_rate = 0.2  # ねむみ摩擦
            initial_r = 3.5
            current_r = initial_r * np.exp(-decay_rate * t)
            omega = 3.0

            x = current_r * np.cos(omega * t)
            y = current_r * np.sin(omega * t)
            # 深さを新しいグリッド定数に合わせる
            z = -DEPTH_FACTOR * np.exp(-0.5 * current_r**2)

            mob.move_to([x, y, z])

        # 数式の表示
        metric_tex = manim.MathTex(
            r"ds^2 = -\left(1 - \frac{R_c}{r}\right) c^2 dt^2 + \dots",
            color=manim.YELLOW,
            font_size=30,
        ).to_corner(manim.UL)

        # テキスト類 (日本語)
        friction_tex = manim.Text(
            "ねむみ摩擦 発動", color=manim.RED, font_size=24
        ).to_corner(manim.UR)

        # 追加: おやすみ軌道のラベル
        orbit_label = (
            manim.Text("おやすみ軌道", color=manim.YELLOW, font_size=24)
            .next_to(metric_tex, manim.DOWN, buff=0.5)
            .to_edge(manim.LEFT)
        )

        self.add_fixed_in_frame_mobjects(metric_tex)
        self.play(manim.Write(metric_tex))

        # アニメーション開始
        human.add_updater(update_human)
        self.add(human)

        # おやすみ軌道のラベルを表示
        self.add_fixed_in_frame_mobjects(orbit_label)
        self.play(manim.FadeIn(orbit_label))

        self.wait(2)
        self.add_fixed_in_frame_mobjects(friction_tex)
        self.play(manim.Write(friction_tex))
        self.play(manim.Indicate(grid, color=manim.RED))

        self.wait(8)

        # 5. 終局
        human.remove_updater(update_human)
        # 日本語のみに変更
        final_text = manim.Text(
            "事象の地平線: 入眠完了", font_size=48, color=manim.WHITE
        )
        self.set_camera_orientation(phi=0 * manim.DEGREES, theta=-90 * manim.DEGREES)
        self.add_fixed_in_frame_mobjects(final_text)
        self.play(
            manim.FadeIn(final_text), manim.FadeOut(grid), manim.FadeOut(ofuton_group)
        )
        self.wait(2)
