import pyglet


# 创建一个透明窗口的函数
def create_transparent_window(width, height, alpha):
    window = pyglet.window.Window(width=width, height=height, style='transparent', visible=False)

    # 设置窗口的背景色为透明
    @window.event
    def on_draw():
        window.clear()

    # 设置窗口的透明度
    @window.event
    def on_activate():
        window.set_vsync(False)
        pyglet.gl.glClearColor(0, 0, 0, 0)

    # 显示窗口
    window.set_vsync(True)
    window.set_exclusive_mouse(False)
    window.set_visible(True)
    return window


# 创建一个宽400像素、高300像素、透明度为0.5的窗口
width, height = 400, 300
alpha = 0.5  # 透明度在0到1之间，0为完全透明，1为完全不透明
window = create_transparent_window(width, height, alpha)

# 运行Pyglet事件循环
pyglet.app.run()