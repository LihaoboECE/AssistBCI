from psychopy import monitors

#这是用来修复primary_monitor文件损坏的文件，请先删除primary_monitor文件
win_size=[1920,1081]
width=59.6
distance=60
mon = monitors.Monitor(
            name="primary_monitor",
            width=width,
            distance=distance,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
            verbose=False,
        )
mon.setSizePix(win_size)  # 显示器的分辨率
mon.save()