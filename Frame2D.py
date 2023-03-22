import Charts as Ch
import pandas as pd
import numpy as np
import math
import os
dpath = os.path.join(os.curdir, 'data')
Fr = pd.read_csv(os.path.join(dpath, 'Frames2D.csv'), sep=';')
Sc = pd.read_csv(os.path.join(dpath, 'Sections2D.csv'), sep=';')


class Frame:
    """КЭ 2D стержня."""

    def setdata(self, f):
        """Данные КЭ"""

        self.fedat = Fr.loc[f, :]

    def tfea(self, fedat):
        """Тип КЭ"""

        sr = self.fedat['Start_RY']
        er = self.fedat['End_RY']
        if sr == 0 and er == 0:
            self.TF = 1
        elif sr == 0 and er == 1:
            self.TF = 2
        elif sr == 1 and er == 0:
            self.TF = 3
        else:
            self.TF = 4

    def mL(self, cosa, sina):
        """Матрица преобразования координат"""

        self.L = np.array([[cosa, sina, 0, 0, 0, 0],
                           [-sina, cosa, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, cosa, sina, 0],
                           [0, 0, 0, -sina, cosa, 0],
                           [0, 0, 0, 0, 0, 1]])

    def mk(self, EA, EI, l):
        """Матрица жесткости КЭ в глобальной системе координат"""
        k = np.zeros((6, 6))
        if self.TF == 1:
            k[0, 0] = EA / l
            k[1, 1] = 12 * EI / l ** 3
            k[2, 2] = 4 * EI / l
            k[3, 3] = k[0, 0]
            k[4, 4] = k[1, 1]
            k[5, 5] = k[2, 2]
            k[0, 3] = -k[0, 0]
            k[1, 2] = -6 * EI / l ** 2
            k[1, 4] = -k[1, 1]
            k[1, 5] = k[1, 2]
            k[2, 4] = -k[1, 2]
            k[2, 5] = 2 * EI / l
            k[4, 5] = k[2, 4]
        elif self.TF == 2:
            k[0, 0] = EA / l
            k[1, 1] = 3 * EI / l ** 3
            k[2, 2] = 3 * EI / l
            k[3, 3] = k[0, 0]
            k[4, 4] = k[1, 1]
            k[0, 3] = -k[0, 0]
            k[1, 2] = -3 * EI / l ** 2
            k[1, 4] = -k[1, 1]
            k[2, 4] = -k[1, 2]
        elif self.TF == 3:
            k[0, 0] = EA / l
            k[1, 1] = 3 * EI / l ** 3
            k[3, 3] = k[0, 0]
            k[4, 4] = k[1, 1]
            k[5, 5] = 3 * EI / l
            k[0, 3] = -k[0, 0]
            k[1, 4] = -k[1, 1]
            k[1, 5] = -3 * EI / l ** 2
            k[4, 5] = -k[1, 5]
        else:
            k[0, 0] = EA / l
            k[3, 3] = k[0, 0]
            k[0, 3] = -k[0, 0]
        for i in range(0, 6):
            for j in range(0, 6):
                k[j, i] = k[i, j]
        self.k = np.matmul(np.matmul(np.transpose(self.L), k), self.L)

    def prop(self, J0, J1):
        """Свойства элемента"""

        x2 = J1.jntdat['X']
        x1 = J0.jntdat['X']
        z2 = J1.jntdat['Z']
        z1 = J0.jntdat['Z']
        sf = self.fedat['Stiffness']
        E = Sc['E'][sf]
        A = Sc['A'][sf]
        I = Sc['I'][sf]
        self.l = math.sqrt((z2 - z1) ** 2 + (x2 - x1) ** 2)  # Длина КЭ
        cosa = (x2 - x1) / self.l
        sina = (z2 - z1) / self.l
        self.EA = E * A  # Жесткость продольная
        self.EI = E * I  # Жесткость изгиба
        self.tfea(self.fedat)
        self.mL(cosa, sina)

    def feloads(self, i, l):
        """Узловые равнодействующие от равномерных нагрузок"""

        q1 = Fr['q1'][i]
        q2 = Fr['q2'][i]
        if self.TF == 1:
            Fq = np.array((q1 * l / 2, q2 * l / 2, -q2 * l **
                           2 / 12, q1 * l / 2, q2 * l / 2, q2 * l ** 2 / 12))
        elif self.TF == 2:
            Fq = np.array((q1 * l / 2, 5 * q2 * l / 8, -q2 * l **
                          2 / 8, q1 * l / 2, 3 * q2 * l / 8, 0))
        elif self.TF == 3:
            Fq = np.array((q1 * l / 2, 3 * q2 * l / 8, 0, q1 *
                          l / 2, 5 * q2 * l / 8, q2 * l ** 2 / 8))
        elif self.TF == 4:
            Fq = np.array((q1 * l / 2, q2 * l / 2, 0,
                          q1 * l / 2, q2 * l / 2, 0))
        else:
            Fq = np.zeros(6)

        self.Fq = np.reshape(np.transpose(Fq), (6, 1))


class Joint:
    """Узел КЭ 2D стержня """

    def __init__(self):
        """Constructor"""

        self.Jn = pd.read_csv(os.path.join(dpath, 'Joints2D.csv'), sep=';')

    def setdata(self, j):
        """Данные узла КЭ"""

        self.jntdat = self.Jn.loc[j, :]


class Model:
    """Расчетная модель"""

    def __init__(self, Joint0, Joint1, Framei):
        """Constructor"""

        self.Joint0 = Joint0
        self.Joint1 = Joint1
        self.Framei = Framei

    def mA(self):
        """Матрица топологии"""

        k = 0
        self.nj = self.Joint0.Jn.shape[0]
        self.nf = Fr.shape[0]
        DoF = np.zeros((self.nj, 3))  # Степени свободы узлов
        for i in range(0, self.nj):
            for j in range(0, 3):
                DoF[i, j] = i + j + k
            k = k + 2
        self.A = np.zeros((6, self.nf))
        for i in range(0, self.nf):
            for j in range(0, 3):
                self.A[j, i] = DoF[Fr['Start'][i]][j]
                self.A[j + 3, i] = DoF[Fr['End'][i]][j]
        self.mC()

    def mC(self):
        """Матрица положения КЭ"""
        self.Ci = np.zeros((self.nf, 6, self.nj * 3))
        for i in range(0, self.nf):
            for j in range(0, 6):
                self.Ci[i, j, int(self.A[j, i])] = 1
        self.mK()

    def mK(self):
        """Матрицы жесткости и нагрузки в глобальной СК"""

        self.Ki = np.zeros((self.nf, 6, 6))
        self.Li = np.zeros((self.nf, 6, 6))
        self.Fqi = np.zeros((self.nf, 6, 1))
        self.Fq = np.zeros((self.nj * 3, 1))
        self.K = np.zeros((self.nj * 3, self.nj * 3))
        self.li = np.zeros(self.nf)
        self.q2i = np.zeros(self.nf)
        for i in range(0, self.nf):
            self.Framei.setdata(i)
            self.Joint0.setdata(self.Framei.fedat['Start'])
            self.Joint1.setdata(self.Framei.fedat['End'])
            self.Framei.prop(self.Joint0, self.Joint1)
            self.li[i] = self.Framei.l
            self.q2i[i] = self.Framei.fedat['q2']
            self.Framei.feloads(i, self.Framei.l)
            self.Fqi[i] = self.Framei.Fq
            self.Fq = self.Fq + \
                np.dot(np.dot(np.transpose(self.Ci[i]),
                              np.transpose(self.Framei.L)), self.Fqi[i])
            self.Framei.mk(self.Framei.EA, self.Framei.EI, self.Framei.l)
            self.Ki[i] = self.Framei.k
            self.Li[i] = self.Framei.L
            self.K = self.K + \
                np.matmul(np.matmul(np.transpose(
                    self.Ci[i]), self.Ki[i]), self.Ci[i])
        self.jntloads()

    def jntloads(self):
        """Узловые нагрузки"""

        self.Fp = np.zeros((self.nj * 3, 1))
        for i in range(0, self.nj):
            self.Fp[i * 3] = self.Joint0.Jn['Fx'][i]
            self.Fp[i * 3 + 1] = self.Joint0.Jn['Fz'][i]
            self.Fp[i * 3 + 2] = self.Joint0.Jn['My'][i]
        self.F = self.Fp + self.Fq

    def rstrs(self):
        """Граничные условия"""
        for i in range(0, self.nj):
            rstr = self.Joint0.Jn[['UX', 'UZ', 'RY']].iloc[i]
            for j in range(0, 3):
                if rstr[j] != 0:
                    rs = i * 3 + j
                    for k in range(0, self.nj * 3):
                        self.K[rs, k] = 0
                        self.K[k, rs] = 0
                    self.K[rs, rs] = 1
                    self.F[rs] = 0
            if self.K[i * 3 + 2, i * 3 + 2] == 0:
                self.K[i * 3 + 2, i * 3 + 2] = 1

    def sol(self):
        """Решение"""
        self.U = np.dot(np.linalg.inv(self.K), self.F)

    def mU(self, i):
        """Усилия в КЭ"""
        self.U_ = np.dot(self.Ci[i], self.U)

    def mS(self):
        self.Si = np.zeros((self.nf, 6, 1))
        for i in range(0, self.nf):
            self.mU(i)
            self.Si[i] = np.dot(
                np.matmul(self.Li[i], self.Ki[i]), self.U_) - self.Fqi[i]

    def nqm(self):
        n = 4
        coord = self.Joint0.Jn.values[:, :2]
        id = Fr.values[:, :2]
        joints = coord[id]
        N = np.zeros((self.nf, 2))
        Q = np.zeros((self.nf, 2))
        M = np.zeros((self.nf, 5))
        for i in range(0, self.nf):
            Mi = np.zeros(n + 1)
            q2 = self.q2i[i]
            S = self.Si
            l = self.li[i]
            Ni = [-S[i][0][0], S[i][3][0]]
            N[i, :] = Ni
            Qi = [-S[i][1][0], S[i][4][0]]
            Q[i, :] = Qi
            Mi[0] = S[i][2][0]
            Mi[n] = -S[i][5][0]
            dM = (Mi[n] - Mi[0]) / l
            dl = l / n
            for j in range(0, n - 1):
                x = (j + 1) * dl
                Mi[j + 1] = -q2 * x * (l - x) / 2 + (Mi[0] + x * dM)
            M[i, :] = Mi
        pd.DataFrame(self.K).to_csv(os.path.join(dpath, 'K.csv'), sep=';')
        print('Перемещения узлов')
        print(pd.DataFrame(np.round(self.U.reshape(self.nj, 3), 6),
                           columns=['UX[м]', 'ZX[м]', 'RX[рад]']))
        pd.DataFrame(self.U.reshape(self.nj, 3),
                     columns=['UX', 'ZX', 'RX']).to_csv(os.path.join(dpath, 'U.csv'), sep=';')
        print('Продольные силы в сечениях элементов, N')
        print(pd.DataFrame(np.round(N, 4),
              columns=['Начало[кН]', 'Конец[кН]']))
        pd.DataFrame(N, columns=['Start', 'End']).to_csv(
            os.path.join(dpath, 'N.csv'), sep=';')
        print('Поперечные силы в сечениях элементов, Q')
        print(pd.DataFrame(np.round(Q, 4),
              columns=['Начало[кН]', 'Конец[кН]']))
        pd.DataFrame(Q, columns=['Start', 'End']).to_csv(
            os.path.join(dpath, 'Q.csv'), sep=';')
        print('Изгибающие моменты в сечениях элементов, M')
        print(pd.DataFrame(np.round(M, 4),
                           columns=['Начало[кН*м]', '0.25*L[кН*м]', '0.5*L[кН*м]', '0.75*L[кН*м]', 'Конец[кН*м]']))
        Ch.geom(coord, joints, 'Геометрия')
        pd.DataFrame(M, columns=['Start', '0.25*L', '0.5*L', '0.75*L', 'End']). \
            to_csv(os.path.join(dpath, 'M.csv'), sep=';')


if __name__ == "__main__":
    Framei = Frame()
    Joint0 = Joint()
    Joint1 = Joint()
    Model1 = Model(Joint0, Joint1, Framei)
    Model1.mA()
    Model1.rstrs()
    Model1.sol()
    Model1.mS()
    Model1.nqm()
