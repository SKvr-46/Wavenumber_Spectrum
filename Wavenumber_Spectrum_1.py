#http://equake-rc.info/SRCMOD/searchmodels/viewmodel/s2016KUMAMO01YAGI/からのすべりのデータ入力を受け取る
#一番下にコピペ用のslipのデータ
#rowは縦のグリッド数(例では10個)
#columnは横のグリッド数(例では28個)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

#準備段階// このinput段落は後に削除を予定
array = input("すべり分布を入力").split()  # スペースで区切られた入力を配列に格納
row =  int(input("行数を入力"))
column = int(input("列数を入力"))
float_array = list(map(float, array))  # 全ての要素をfloatに変換
two_d_array = []
for i in range(row):
    row = float_array[i*column:(i+1)*column]  # column(28)個の要素をスライスして取得
    two_d_array.append(row)
# print(two_d_array)
print(len(two_d_array))
print(len(two_d_array[0]))

#入力されたすべりのデータ
slip = np.array(two_d_array)

#実験的に任意の2次元断層すべり分布を生成する以外では、1つ上のセルですべりデータを受け取って、
#two_d_arrayを作る必要がある。




# 2次元断層すべりを生成する 幅56km,深さ20km
#データは10行18列
nx = 28 #サンプリング数
ny = 10
dx=2 #サンプリング間隔（dxはグリッドの横の長さ）
dy=2


#波数スペクトル行列の行（列）で。波数0が何番目なのかを探索する関数（必須ではない）
def judge(x):
    if(x % 2 == 0):
        return int(x / 2) 
    elif(x % 2 != 0):
        return int((x - 1) / 2)
    
# x軸方向の波数スペクトルを計算する
kx =2*np.pi* np.fft.fftfreq(slip.shape[1], d=dx) #データ点数28個
modified_kx = np.fft.fftshift(kx)    #波数の列を小さい順にする
kx_spec =np.abs(np.fft.fftshift(np.fft.fft2(slip))) #振幅

# y軸方向の波数スペクトルを計算する
ky = 2*np.pi*np.fft.fftfreq(slip.shape[0], d=dx) #データ点数10個
modified_ky = np.fft.fftshift(ky)    #波数の列を小さい順にする
ky_spec =np.abs(np.fft.fftshift(np.fft.fft2(slip))) #振幅


#出力する
fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(15, 5))
ax1.plot(modified_kx, kx_spec[judge(ny),:], 'b.-')  
ax1.set_xlabel('kx')
ax1.set_ylabel('amplitude')
ax1.set_title('Wavenumber spectrum along x-axis')
ax1.set_xlim(0, max(modified_kx))

ax2.plot(modified_ky, ky_spec[:,judge(nx)], 'r.-')  
ax2.set_xlabel('ky')
ax2.set_ylabel('amplitude')
ax2.set_title('Wavenumber spectrum along y-axis')
ax2.set_xlim(0, max(modified_ky))


# 2次元波数スペクトル（の絶対値の自然対数の分布）を描画する
im = ax3.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(slip)))), cmap='jet', origin="lower",extent=(min(kx), max(kx), min(ky), max(ky)))
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
ax3.set_title('2D Wavenumber spectrum')
fig.colorbar(im, ax=ax3)
im.autoscale()
plt.show()

#すべり分布の出力
plt.imshow(slip, cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title("slip")
plt.show()


#波数0で1に規格化された縦方向の波数スペクトルと、横方向の波数スペクトルの出力
#kx_specの規格化：kx=0の列で全体を割る。
#こうすることで、どのkyの場合でも、始まりが1になる。
kx_spec_normalized = kx_spec / kx_spec[:,judge(nx)][:, np.newaxis] 
plt.plot(modified_kx,kx_spec_normalized[judge(ny),:], 'b.-')  

#ky_specの規格化：ky=0の行で割る
ky_spec_normalized = ky_spec / ky_spec[judge(ny),:]
plt.plot(modified_ky, ky_spec_normalized[:,judge(nx)], 'r.-')  

plt.xlabel('wavenumber')
plt.ylabel('amplitude')
plt.title('Normalized Wavenumber spectrum ')
plt.xlim(0, max(max(modified_ky), max(modified_kx)))
plt.show()


#フーリエ変換された2次元波数スペクトル（の絶対値の自然対数の分布）をcsvに保存する
#上のプログラムの2D Wavenumber spectrumが、この表をヒートマップにしたもの
TWS = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(slip))))
twoD_spec_data = pd.DataFrame(TWS, index=modified_ky, columns=modified_kx)
twoD_spec_data.index.name = 'Ky'
twoD_spec_data.columns.name = 'Kx'
twoD_spec_data.to_csv("twoD_spec_data.csv")
twoD_spec_data


#波数と振幅の関係
#波数成分(ky[i],kx[j])に属する振幅が、kx_spec[i,j]で表される。
spec_data = pd.DataFrame(kx_spec, index=modified_ky, columns=modified_kx)
spec_data.index.name = 'Ky'
spec_data.columns.name = 'Kx'
spec_data.to_csv("spec_data.csv")
spec_data


if __name__ == "__main__":
    main()