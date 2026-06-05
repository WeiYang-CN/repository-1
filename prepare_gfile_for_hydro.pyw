import numpy as np
import re
from scipy.interpolate import RegularGridInterpolator, interp1d
import os

# ================== 稳健的 gfile 解析 ==================
def read_gfile(filename):
    def extract_floats(line):
        return [float(x) for x in re.findall(r'[+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?', line)]

    with open(filename, 'r') as f:
        f.read(50) # 跳过前50字符，这些行不是需要的。只剩下第一行的 3 129 129 需要读取
        nw, nh = None, None
        while nw is None or nh is None:
            line = f.readline() # 读取第一行，3 129 129
            if not line: break  # 如果读取到的 line 是空的（也就是没有内容），就立即退出循环
            nums = extract_floats(line) # 提取3个数字，[3.0, 129.0, 129.0]，nw=129, nh=129
            if len(nums) >= 3:
                nw = int(nums[1]); nh = int(nums[2]) # R 方向的网格数 nw = 129，Z 方向的网格数 nh = 129
        print(f"gfile 网格: nw={nw}, nh={nh}")

        params = [] # 参数列表，20个参数
        while len(params) < 20: # 读取20个参数, 正好对应 gfile 的第 2 行到第 5 行
            line = f.readline()
            if not line: break
            params.extend(extract_floats(line)) # 先将当前行 line 中的所有浮点数提取出来，然后将这些浮点数逐一添加到列表 params 的末尾
        params = params[:20]
        rdim   = params[0]; zdim   = params[1]; rcentr = params[2]
        rleft  = params[3]; zmid   = params[4]
        rmaxis = params[5]; zmaxis = params[6]
        simag  = params[7]; sibry  = params[8]
        bcentr = params[9]; current= params[10]

        def read_1d(n): # 顺序读入大小为 n（这里 n = nw = 129）的物理剖面
            data = []
            while len(data) < n: # 每行排列 5 个数据。 因此，每个 129 长度的数组，在文件中占用 129 / 5 = 25.8 行，即 26 行
                line = f.readline()
                if not line: break
                data.extend(extract_floats(line))
            return np.array(data[:n])

        fpol   = read_1d(nw) # 极向电流函数 $F(\psi) = R B_\phi$
        pres   = read_1d(nw) # 等离子体压强 $P(\psi)$
        ffprim = read_1d(nw) # $F dF/d\psi$ 函数
        pprime = read_1d(nw) # 压强梯度 $dP/d\psi$
        # 二维极向磁通 $\psi(R, Z)$ 网格分布. 包含 $129 \times 129 = 16641$ 个数据的大矩阵, 在文件中占用 $16641 / 5 = 3328.2 \rightarrow$ 3329 行
        psirz  = read_1d(nw * nh).reshape(nw, nh) 
        qpsi   = read_1d(nw) # 安全因子 $q(\psi)$ 剖面

        temp = []
        while len(temp) < 2:
            line = f.readline()
            if not line: break
            temp.extend(extract_floats(line))
        # 描述整个托卡马克的物理边界几何
        nbbbs  = int(temp[0]) # 等离子体边界坐标. 377*2=754 个数据. 顺时针或逆时针环绕等离子体最后闭合磁面的 $(R_1, Z_1, R_2, Z_2, \dots)$ 真实物理坐标对
        nlimtr = int(temp[1]) # 限制器边界坐标. 12*2=24 个数据. 顺时针或逆时针环绕限制器最后闭合磁面的 $(R_1, Z_1, R_2, Z_2, \dots)$ 真实物理坐标对

        def read_pairs(n):
            data = read_1d(2 * n)
            return data.reshape(n, 2)

        rbbbs_zbbbs = read_pairs(nbbbs)
        rlim_zlim   = read_pairs(nlimtr)
        rbbbs = rbbbs_zbbbs[:,0]; zbbbs = rbbbs_zbbbs[:,1]
        rlim  = rlim_zlim[:,0];   zlim  = rlim_zlim[:,1]

    return {
        'nw': nw, 'nh': nh,
        'rdim': rdim, 'zdim': zdim, 'rleft': rleft, 'zmid': zmid,
        'rmaxis': rmaxis, 'zmaxis': zmaxis,
        'simag': simag, 'sibry': sibry, 'bcentr': bcentr, 'current': current,
        'fpol': fpol, 'pres': pres, 'ffprim': ffprim, 'pprime': pprime,
        'psirz': psirz, 'qpsi': qpsi,
        'rbbbs': rbbbs, 'zbbbs': zbbbs, 'rlim': rlim, 'zlim': zlim
    }

# ================== 主程序 ==================
print("解析 gfile ...")
g = read_gfile('gfile.txt')
print("解析成功。")

# 原始 gfile 网格 (R, Z)
r_orig = np.linspace(g['rleft'], g['rleft'] + g['rdim'], g['nw']) # $0.1 \rightarrow (0.1 + 1.4) = 1.5$ 米，平分成 129 个网格点
z_orig = np.linspace(g['zmid'] - g['zdim']/2, g['zmid'] + g['zdim']/2, g['nh']) # $-1.7 \rightarrow 1.7$ 米，平分成 129 个网格点
Rmesh_orig, Zmesh_orig = np.meshgrid(r_orig, z_orig)   # (nh, nw) 对应 (Z, R)
psirz = g['psirz'].T   # 转置为 (nh, nw) 对应 (Z, R)

# 计算极向场 BR, BZ

# 计算二维网格上的磁通梯度
dpsi_dR = np.gradient(psirz, r_orig, axis=1) # $\partial\psi/\partial R$
dpsi_dZ = np.gradient(psirz, z_orig, axis=0) # $\partial\psi/\partial Z$

BR_orig = -1.0 / (2 * np.pi * Rmesh_orig) * dpsi_dZ # 磁场分量 $B_R = -1 / (2 \pi R) \frac{\partial \psi}{\partial Z}$
BZ_orig =  1.0 / (2 * np.pi * Rmesh_orig) * dpsi_dR # 磁场分量 $B_Z = 1 / (2 \pi R) \frac{\partial \psi}{\partial R}$

# 环向场 Bphi $B_\phi = F(\psi) / R$
psi_1d = np.linspace(g['simag'], g['sibry'], g['nw']) # 磁轴处极向磁通 simag，边缘处极向磁通 sibry, 平分成 129 个网格点
F_interp = interp1d(psi_1d, g['fpol'], kind='linear', bounds_error=False, fill_value='extrapolate') # [simag, sibry] 范围内通过插值计算任意半径处的磁通
F_orig = F_interp(psirz) # 从磁轴到边缘任意处磁通 F(psi)的值
Bphi_orig = F_orig / Rmesh_orig # 环向场 Bphi

# 建立 (R, Z) 插值器（输入顺序：Z, R）, 从离散的网格空间到连续物理空间的数值映射
interp_BR   = RegularGridInterpolator((z_orig, r_orig), BR_orig,   bounds_error=False, fill_value=0.0)
interp_BZ   = RegularGridInterpolator((z_orig, r_orig), BZ_orig,   bounds_error=False, fill_value=0.0)
interp_Bphi = RegularGridInterpolator((z_orig, r_orig), Bphi_orig, bounds_error=False, fill_value=0.0)

# 新网格定义（笛卡尔）
X, Y, Z = 100, 100, 600
x_arr = np.linspace(0.1, 1.5, X)      # 径向 0.1 → 1.5 m
y_arr = np.linspace(-0.01, 0.01, Y)    # 环向薄片 ±1 cm
z_arr = np.linspace(-1.7, 1.7, Z)      # 垂直 -1.7 → 1.7 m

# 预分配数组 (Z, X, Y)
Bx_field = np.zeros((Z, X, Y), dtype=np.float32)
By_field = np.zeros((Z, X, Y), dtype=np.float32)
Bz_field = np.zeros((Z, X, Y), dtype=np.float32)

print("开始生成磁场（线性插值）...")
# 遍历所有网格点
for iz, z in enumerate(z_arr): # 读取$Z$ 轴方向上的网格坐标点, 返回
    if iz % 50 == 0:
        print(f"  处理 Z 层 {iz}/{Z}")
    for ix, x in enumerate(x_arr):
        for iy, y in enumerate(y_arr):
            R = np.sqrt(x**2 + y**2) # 径向距离 
            phi = np.arctan2(y, x) # 环向角度
            point = np.array([z, R])   # (Z, R)
            br   = interp_BR(point)[0]
            bz   = interp_BZ(point)[0]
            bphi = interp_Bphi(point)[0]
            # 笛卡尔分量
            cos_p = np.cos(phi)
            sin_p = np.sin(phi)
            Bx = br * cos_p - bphi * sin_p
            By = br * sin_p + bphi * cos_p
            Bz = bz
            # 不同网格索引对应的磁场在直角坐标下分量
            Bx_field[iz, ix, iy] = Bx
            By_field[iz, ix, iy] = By
            Bz_field[iz, ix, iy] = Bz

# 写入 bfield.bin（外层 Z，中层 X，内层 Y）
print("写入 bfield.bin ...")
with open('bfield.bin', 'wb') as f:
    for iz in range(Z):
        for ix in range(X):
            for iy in range(Y):
                f.write(Bx_field[iz, ix, iy].tobytes())
                f.write(By_field[iz, ix, iy].tobytes())
                f.write(Bz_field[iz, ix, iy].tobytes())
print(f"bfield.bin 已生成，大小: {os.path.getsize('bfield.bin')} 字节")

# 写入 flowfield.bin（均匀静态等离子体）
print("写入 flowfield.bin ...")
rho = 33.0e3; p0 = 1.2e16
vx = vy = vz = 0.0
B0x = B0y = B0z = 0.0
with open('flowfield.bin', 'wb') as f:
    for iz in range(Z):
        for ix in range(X):
            for iy in range(Y):
                f.write(np.float32(rho).tobytes())
                f.write(np.float32(vx).tobytes())
                f.write(np.float32(vy).tobytes())
                f.write(np.float32(vz).tobytes())
                f.write(np.float32(p0).tobytes())
                f.write(np.float32(B0x).tobytes())
                f.write(np.float32(B0y).tobytes())
                f.write(np.float32(B0z).tobytes())
print(f"flowfield.bin 已生成，大小: {os.path.getsize('flowfield.bin')} 字节")

print("所有文件生成完毕。")