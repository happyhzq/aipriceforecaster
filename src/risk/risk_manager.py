import numpy as np

def kelly_fraction(p_win, b=1.0, conservative=0.5):
    q = 1 - p_win
    f = ((b*p_win - q)/b)
    return np.clip(conservative * f, 0.0, 1.0)

def apply_limits(position, max_leverage=1.0, max_pos=1.0):
    position = np.clip(position, -max_pos, max_pos)
    # 可加入账户净值与杠杆约束，这里简化
    return position