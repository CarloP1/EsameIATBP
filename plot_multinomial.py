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

PP = np.linspace(0, 1, 10)

def likelihood(x, p, n):
    '''
    likelihood,
    vedere docu di multinomial
    '''
    return stat.multinomial.pmf(x=x, n=n, p=p)

def prior():
    '''
    Prior
    '''
    return 2/((N+1)*(N+2)-2)

P = np.zeros((N+1, N+1, len(PP))) #Matrice che conterrà La postirior



for i in range(N+1):
    print(f'{(i+1)/(N+1) *100:.1f}% \r', end='')
    for j in range(N+1):
        # vincolo di elezioni non truccate: votano solo N persone
        NC = N - Na[i] - Nb[j]
        for k in range(len(PP)):
            pa = Na[i]*PP[k]/N+(N-Na[i])*(1-PP[k])/(2*N)
            pb = Nb[j]*PP[k]/N+(N-Nb[j])*(1-PP[k])/(2*N)
            pc = NC*PP[k]/N+(N-NC)*(1-PP[k])/(2*N)
            if NC > 0:
                P[i, j, k] = likelihood(x=[na, nb, nc], p=[pa, pb, pc], n=na+nb+nc)*prior()

pVA = 0
pVB = 0
gridx, gridy = np.meshgrid(Na, Nb)
Norm_P = np.zeros(len(PP))
Norm_Pa = np.zeros(len(PP))
Norm_Pb = np.zeros(len(PP))
P_A = np.zeros((N+1, len(PP)))
P_B = np.zeros((N+1, len(PP)))
pVA = np.zeros(len(PP))
pVB = np.zeros(len(PP))

for k in range(len(PP)):
    for i in range(N+1):
        for j in range(N+1):
            if i>j and i>(N-j)/2:
                pVA[k] = pVA[k] + P[i, j, k]
            if j>i and j>(N-i)/2:
                pVB[k] = pVB[k] + P[i, j, k]

#dato che il passo degli N è unitario basta sommare
for k in range(len(PP)):
    Norm_P[k] = sum(sum(P[:,:,k])) #sommo due volte, per sommare su tutti gli indici
    P[:,:,k] /= Norm_P[k]
    #Marginalizzo su Na
    P_A[:, k] = np.array([sum(P[i, :, k]) for i in range(N+1)])
    Norm_Pa[k] = sum(P_A[:,k])
    P_A[:, k] /= Norm_Pa[k]
    #Marginalizzo su Nb
    P_B[:, k] = np.array([sum(P[:, i, k]) for i in range(N+1)])
    Norm_Pb[k] = sum(P_B[:, k])
    P_B[:, k] /= Norm_Pb[k]
    pVA[k] = pVA[k]/Norm_P[k]
    pVB[k] = pVB[k]/Norm_P[k]

print(f'logZ = {np.log(Norm_P)}')

for k in range(len(PP)):
    plt.figure(k)
    plt.title('Posterior distribution')
    levels = np.linspace(0, np.max(P[:,:,k]), 100)
    c=plt.contourf(gridx, gridy, P[:,:,k].T, levels=levels, cmap='gist_ncar')
    plt.colorbar(c)
    plt.title(rf"P($N_A$, $N_B$ |$n_A$={na}, $n_B$={nb}, $n_C$={nc}) with N={N} p={PP[k]:.3f}")
    plt.xlabel(r"$N_A$")
    plt.ylabel(r"$N_B$")

    plt.figure(len(PP)+ 1)
    plt.title(f'Marginalized distributions P($N_A$ |{na}, {nb}, {nc}) with N={N}')
    plt.plot(Na, P_A[:, k], label=rf"p={PP[k]:.1f}")
    plt.xlabel(r"$N_A$")
    plt.ylabel(r"P($N_A$ |$n_A$, $n_B$, $n_C$)")
    plt.legend()
    plt.grid()

    plt.figure(len(PP)+ 2)
    plt.title(f'Marginalized distributions P($N_B$ |{na}, {nb}, {nc}) with N={N}')
    plt.plot(Na, P_B[:, k], label=rf"p={PP[k]:.1f}")
    plt.xlabel(r"$N_B$")
    plt.ylabel(r"P($N_B$ |$n_A$, $n_B$, $n_C$)")
    plt.legend()
    plt.grid()

plt.figure(2*len(PP)+1)
plt.title(f'Probability of victory as a function of p')
plt.plot(PP, pVA, 'ro', label=rf"P($V_A$)")
plt.plot(PP, pVB, 'go', label=rf"P($V_B$)")
plt.xlabel(r"$p$")
plt.ylabel(r"P($V_X$)")
plt.legend()
plt.grid()


plt.show()

end=time.time()-start
print(f'time:{end}')
