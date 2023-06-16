#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[48]:


T = np.linspace(50, 500, 100)


# In[49]:


c = 4.02 - 1.839 * T * 10 ** (-3) + T ** 2 * 3.071 * 10 ** (-5) - T ** 3 * 3.549 * 10 ** (-8)


# In[50]:


# グラフの作成
plt.plot(T, c)
plt.xlabel('T')
plt.ylabel('c')
plt.title('c vs T')
plt.grid(True)
plt.show()


# In[52]:


import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# Tの範囲を指定
T = np.linspace(50, 500, 100)

# Tをシンボルとして定義
T_sym = sp.Symbol('T')

c = 4.02 - 1.839 * T * 10 ** (-3) + T ** 2 * 3.071 * 10 ** (-5) - T ** 3 * 3.549 * 10 ** (-8)

# 式を定義
c_sub = 4.02 - 1.839 * T_sym * 10 ** (-3) + T_sym ** 2 * 3.071 * 10 ** (-5) - 3.549 * 10 ** (-8) * T_sym ** 3

# Tで微分
derivative = sp.diff(c_sub, T_sym)

# 数値に変換してUを計算
U = []
for t in T:
    c_val = c_sub.subs(T_sym, t).evalf()
    derivative_val = derivative.subs(T_sym, t).evalf()
    u_val = c_val / (1 + (t / c_val) * derivative_val)
    U.append(u_val)

    
#グラフを作成
plt.plot(T, c, label='phase')
plt.plot(T, U, label='group')
plt.xlabel('T')
plt.ylabel('Velocity')
plt.title('c and U vs T')
plt.legend()
plt.grid(True)
plt.show()

print(U)


# In[179]:


#cosの単純な合成波を書く
import numpy as np
import matplotlib.pyplot as plt

# 周波数範囲と周波数間隔の設定
start_freq = 0.002  # 開始周波数
end_freq = 0.02  # 終了周波数
freq_step = 0.0002  # 周波数間隔
freqs = np.arange(start_freq, end_freq, freq_step)

# 時間軸の設定
duration = 6000  # 時間の長さ（秒）
t = np.arange(0, duration, 5)  # 時間軸

synthetic_cos_wave = 0
for i in range(len(freqs)):
    synthetic_cos_wave += np.cos(2 * np.pi * freqs[i] * t)
    
plt.plot(t, synthetic_cos_wave)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sum of Cosine Functions at 0 distance')
plt.grid(True)
plt.yticks(range(-40, 100, 10))
plt.show()


# In[98]:


len(freqs)


# In[56]:


#cosの単純な合成波を書く 　30度違い
import numpy as np
import matplotlib.pyplot as plt

# 周波数範囲と周波数間隔の設定
start_freq = 0.002  # 開始周波数
end_freq = 0.02  # 終了周波数
freq_step = 0.0002  # 周波数間隔
freqs = np.arange(start_freq, end_freq, freq_step)

# 時間軸の設定
duration = 5400  # 時間の長さ（秒）
t = np.arange(0, duration, 5)  # 時間軸

x = 2 * np.pi * 12800 /12

synthetic_cos_wave = 0
for i in range(len(freqs)):
    c = 4.02 - 1.839 * (1/freqs[i]) * 10 ** (-3) + (1/freqs[i]) ** 2 * 3.071 * 10 ** (-5) -  3.549 * 10 ** (-8) * (1/freqs[i]) ** 3
    synthetic_cos_wave += np.cos(2*np.pi *  freqs[i] * (x/c - t))
    
plt.plot(t, synthetic_cos_wave)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sum of Cosine Functions at 30 degree distance')
plt.grid(True)
plt.yticks(range(-40, 100, 10))
plt.show()


# In[82]:


#cosの単純な合成波を書く 　90度違い
import numpy as np
import matplotlib.pyplot as plt

# 周波数範囲と周波数間隔の設定
start_freq = 0.002  # 開始周波数
end_freq = 0.02  # 終了周波数
freq_step = 0.0002  # 周波数間隔
freqs = np.arange(start_freq, end_freq, freq_step)

# 時間軸の設定
duration = 5400  # 時間の長さ（秒）
t = np.arange(0, duration, 5)  # 時間軸

x = 2 * np.pi * 12800 /4

synthetic_cos_wave = 0
for i in range(len(freqs)):
    c = 4.02 - 1.839 * (1/freqs[i])* 10 ** (-3) + (1/freqs[i]) ** 2 * 3.071 * 10 ** (-5) - (1/freqs[i]) ** 3 * 3.549 * 10 ** (-8)
    synthetic_cos_wave += np.cos(2*np.pi *  freqs[i] * (x/c - t))
    
plt.plot(t, synthetic_cos_wave)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sum of Cosine Functions at 90 degree distance')
plt.grid(True)
plt.yticks(range(-40, 100, 10))
plt.show()


# In[83]:


#cosの単純な合成波を書く 　150度違い
import numpy as np
import matplotlib.pyplot as plt

# 周波数範囲と周波数間隔の設定
start_freq = 0.002  # 開始周波数
end_freq = 0.02  # 終了周波数
freq_step = 0.0002  # 周波数間隔
freqs = np.arange(start_freq, end_freq, freq_step)

# 時間軸の設定
duration = 5400  # 時間の長さ（秒）
t = np.arange(0, duration, 5)  # 時間軸

x = 2 * np.pi * 12800 * 5 / 12

synthetic_cos_wave = 0
for i in range(len(freqs)): #len(freqs)
    c = 4.02 - 1.839 * (1/freqs[i])* 10 ** (-3) + (1/freqs[i]) ** 2 * 3.071 * 10 ** (-5) - (1/freqs[i]) ** 3 * 3.549 * 10 ** (-8)
    synthetic_cos_wave += np.cos(2*np.pi * freqs[i] * (x/c - t))
    

plt.plot(t, synthetic_cos_wave)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sum of Cosine Functions at 150 degree distance')
plt.grid(True)
plt.yticks(range(-40, 100, 10))

plt.show()


# In[84]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# 周波数範囲と周波数間隔の設定
start_freq = 0.002  # 開始周波数
end_freq = 0.02  # 終了周波数
freq_step = 0.0002  # 周波数間隔
freqs = np.arange(start_freq, end_freq, freq_step)

# 時間軸の設定
duration = 5400  # 時間の長さ（秒）
t = np.arange(0, duration, 5)  # 時間軸

x = 2 * np.pi * 12800 * 5 / 12

synthetic_cos_wave = 0
for i in range(len(freqs)):
    c = 4.02 - 1.839 * (1/freqs[i])* 10 ** (-3) + (1/freqs[i]) ** 2 * 3.071 * 10 ** (-5) - (1/freqs[i]) ** 3 * 3.549 * 10 ** (-8)
    synthetic_cos_wave += np.cos(2*np.pi * freqs[i] * (x/c - t))

# 包絡線の計算
analytic_signal = hilbert(synthetic_cos_wave)
envelope = np.abs(analytic_signal)

# グラフの描画
plt.plot(t, synthetic_cos_wave, label='Signal')
plt.plot(t, envelope, label='Envelope', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Envelope Plot using Hilbert Transform')
plt.legend()
plt.grid(True)
plt.show()


# In[85]:


dis = 2 * np.pi * 12800 * 5 / 12 
dis


# In[79]:


dis / 5.38498936366071


# In[80]:


6222.913245553906 / 60


# In[73]:


#5400 => 1.5h,  6222 => 1.66h


# In[ ]:




