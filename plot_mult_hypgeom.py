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

P = np.zeros((N+1, N+1)) #Matrice che conterrà La postitior

for i in range(N+1):
    print(f'{(i+1)/(N+1) *100:.1f}% \r', end='')
    for j in range(N+1):
        # vincolo di elezioni non truccate: votano solo N persone
        NC = N - Na[i] - Nb[j]
        if NC > 0:
            P[i, j] = likelihood(x=[na, nb, nc], m=[Na[i], Nb[j], NC], n=na+nb+nc)*prior()

pVA = 0
pVB = 0

for i in range(N+1):
    for j in range(N+1):
        if i>j and i>(N-j)/2:
            pVA = pVA + P[i, j]
        if j>i and j>(N-i)/2:
            pVB = pVB + P[i, j]

#dato che il passo degli N è unitario basta sommare
Norm_P = sum(sum(P)) #sommo due volte, per sommare su tutti gli indici
print()
P /= Norm_P
#Marginalizzo su Na
P_A = np.array([sum(P[i, :]) for i in range(N+1)])
Norm_Pa = sum(P_A)
P_A /= Norm_Pa
#Marginalizzo su Nb
P_B = np.array([sum(P[:, i]) for i in range(N+1)])
Norm_Pb = sum(P_B)
P_B /= Norm_Pb



pVA = pVA/Norm_P
pVB = pVB/Norm_P

print('P(VA)=', pVA)
print('P(VB)=', pVB)
print('P(VC)=', 1-pVA-pVB)
print('Evidence=', np.log(Norm_P))

gridx, gridy = np.meshgrid(Na, Nb)
fig = plt.figure(1)
ax1 = fig.add_subplot(projection='3d')
ax1.plot_surface(gridx, gridy, P)
ax1.set_xlabel("Na")
ax1.set_ylabel("Nb")
ax1.set_zlabel("P(Na, Nb | na, nb, nc)")

plt.figure(2)
levels = np.linspace(0, np.max(P), 100)
c=plt.contourf(gridx, gridy, P.T, levels=levels, cmap='gnuplot')
plt.colorbar(c)
plt.title(rf"P($N_A$,$N_B$|$n_A$={na},$n_B$={nb},$n_C$={nc}) with N={N}", fontsize=14)
plt.xlabel(r"$N_A$", fontsize=14)
plt.ylabel(r"$N_B$", fontsize=14)

plt.figure(3)
plt.title(f'Marginalized distributions with N={N}')
plt.plot(Na, P_A, 'r.', label=rf"P($N_A$ |{na}, {nb}, {nc})")
plt.plot(Nb, P_B, 'g.', label=rf"P($N_B$ |{na}, {nb}, {nc})")
plt.xlabel(r"$N_X$", fontsize=14)
plt.ylabel(r"P($N_X$ |$n_A$, $n_B$, $n_C$)", fontsize=14)
plt.legend()
plt.grid()

plt.figure(4)
plt.plot(Nb, P_B)
plt.xlabel("Nb")
plt.ylabel("P(Nb | na, nb, nc)")
plt.grid()


plt.show()

end=time.time()-start
print(f'time:{end}')