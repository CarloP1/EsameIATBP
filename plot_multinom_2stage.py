import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import time

start=time.time()

N = 1000 # Popolazione totale avente diritto di voto

Na = np.linspace(0, N, N+1, dtype=int) # possibili valori
Nb = np.linspace(0, N, N+1, dtype=int) # che le variabili
Nc = np.linspace(0, N, N+1, dtype=int) # possono assumere

na = 20 # risultato
nb = 10 # dei
nc = 5 # sondaggi

p1 = 0.6 # risultato
p2 = 0.4 # dei
p3 = 0.3 # sondaggi

n2aC = np.linspace(na, na+nc, nc+1, dtype=int)
n2aB = np.linspace(na, na+nb, nb+1, dtype=int)

PP = np.linspace(0, 1, 2)

def likelihood(x, m, n):
    '''
    likelihood,
    vedere docu di multivariate_hypergeom
    '''
    return stat.multivariate_hypergeom.pmf(x=x, m=m, n=n)

def prior():
    '''
    Prior
    '''
    return 2/((N+1)*(N+2)-2)

P = np.zeros((N+1, N+1, len(PP))) #Matrice che conterrà La postirior
P2C = np.zeros((N+1,nc+1))
P2B = np.zeros((N+1,nb+1))
Pn2C = np.zeros(nc+1)
Pn2B = np.zeros(nb+1)


for i in range(N+1):
    print(f'{(i+1)/(N+1) *100:.1f}% \r', end='')
    for j in range(nc+1):
        P2C[i, j] = stat.hypergeom.pmf(k=n2aC[j], M=N, N=na+nb+nc, n=Na[i])/N
        Pn2C[j] = stat.binom.pmf(k=n2aC[j]-na, n=nc, p=p3)/nc

for i in range(N+1):
    print(f'{(i+1)/(N+1) *100:.1f}% \r', end='')
    for j in range(nb+1):
        P2B[i, j] = stat.hypergeom.pmf(k=n2aB[j], M=N, N=na+nb+nc, n=Na[i])/N
        Pn2B[j] = stat.binom.pmf(k=n2aB[j]-na, n=nb, p=1-p2)/nb

for i in range(N+1):
    print(f'{(i+1)/(N+1) *100:.1f}% \r', end='')
    for j in range(N+1):
        # vincolo di elezioni non truccate: votano solo N persone
        NC = N - Na[i] - Nb[j]
        if NC > 0:
            P[i, j] = likelihood(x=[na, nb, nc], m=[Na[i], Nb[j], NC], n=na+nb+nc)*prior()

pVA = 0
pVB = 0
pLC = 0
pLB = 0
pLA = 0
pCVA2 = 0
pBVA2 = 0

gridx, gridy = np.meshgrid(Na, Nb)
gridxC, gridyC = np.meshgrid(Na, n2aC)
gridxB, gridyB = np.meshgrid(Na, n2aB)

for i in range(N+1):
    for j in range(N+1):
        if i>j and i>(N-j)/2:
            pVA = pVA + P[i, j]
        if j>i and j>(N-i)/2:
            pVB = pVB + P[i, j]
        if j<i and j<(N-i-j):
            pLB = pLB + P[i, j]
        if (N-i-j)<i and (N-i-j)<j:
            pLC = pLC + P[i, j]
    for l in range(nc+1):
        if i>N/2:
            pCVA2 = pCVA2 + P2C[i, l]*Pn2C[l]
    for k in range(nb+1):
        if i>N/2:
            pBVA2 = pBVA2 + P2B[i, k]*Pn2B[k]

#dato che il passo degli N è unitario basta sommare
Norm_P = sum(sum(P))
P /= Norm_P
Norm_PC = 0
Norm_PB = 0
for i in range(N+1):
    for j in range(nc+1):
        Norm_PC = Norm_PC + P2C[i, j]*Pn2C[j]
    for k in range(nb+1):
        Norm_PB = Norm_PB + P2B[i, k]*Pn2B[k]

Norm_Pn2C = sum(Pn2C)
Norm_Pn2B = sum(Pn2B)
Norm_P2C = sum(sum(P2C))
Norm_P2B = sum(sum(P2B))
P2C /= Norm_P2C
P2B /= Norm_P2B #sommo due volte, per sommare su tutti gli indici
Pn2C /= Norm_Pn2C
Pn2B /= Norm_Pn2B
pVA = pVA/Norm_P
pVB = pVB/Norm_P
pLB = pLB/Norm_P
pLC = pLC/Norm_P
pCVA2 = pCVA2/(Norm_PC)
pBVA2 = pBVA2/(Norm_PB)


pVA2 = pBVA2*pLB + pCVA2*pLC

print('P(VA)=', pVA)
print('P(VB)=', pVB)
print('P(VC)=', 1-pVA-pVB)
print('P(LB)=', pLB)
print('P(LC)=', pLC)
print('P(VAC)=', pCVA2)
print('P(VAB)=', pBVA2)
print('P(VA2)=', pVA2)

end=time.time()-start
print(f'time:{end}')