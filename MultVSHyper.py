import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import time

start=time.time()

NN = [100, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000]
diff = np.zeros(len(NN))
diffmax = np.zeros(len(NN))

for k, N in enumerate(NN):
    #N = 1000 # Popolazione totale avente diritto di voto

    Na = np.linspace(0, N, N+1, dtype=int) # possibili valori
    Nb = np.linspace(0, N, N+1, dtype=int) # che le variabili
    Nc = np.linspace(0, N, N+1, dtype=int) # possono assumere

    na = 20 # risultato
    nb = 10 # dei
    nc = 5 # sondaggi

    PP = 1

    def likelihood(x, p, n):
        '''
        likelihood,
        vedere docu di multivariate_hypergeom
        '''
        return stat.multinomial.pmf(x=x, n=n, p=p)

    def prior():
        '''
        Prior
        '''
        return 2/((N+1)*(N+2)-2)

    P = np.zeros((N+1, N+1)) #Matrice che conterrà La postirior
    Ph = np.zeros((N+1, N+1))


    for i in range(N+1):
        print(f'{(i+1)/(N+1) *100:.1f}% \r', end='')
        for j in range(N+1):
            # vincolo di elezioni non truccate: votano solo N persone
            NC = N - Na[i] - Nb[j]
            pa = Na[i]*PP/N+(N-Na[i])*(1-PP)/(2*N)
            pb = Nb[j]*PP/N+(N-Nb[j])*(1-PP)/(2*N)
            pc = NC*PP/N+(N-NC)*(1-PP)/(2*N)
            if NC > 0:
                P[i, j] = likelihood(x=[na, nb, nc], p=[pa, pb, pc], n=na+nb+nc)*prior()
                Ph[i, j] = stat.multivariate_hypergeom.pmf(x=[na, nb, nc], m=[Na[i], Nb[j], NC], n=na+nb+nc)*prior()



    #dato che il passo degli N è unitario basta sommare

    Norm_P = sum(sum(P)) #sommo due volte, per sommare su tutti gli indici
    P /= Norm_P
    #Marginalizzo su Na
    P_A = np.array([sum(P[i, :]) for i in range(N+1)])
    Norm_Pa = sum(P_A)
    P_A /= Norm_Pa

    Norm_Ph = sum(sum(Ph))
    Ph /= Norm_Ph
    P_Ah = np.array([sum(Ph[i, :]) for i in range(N+1)])
    Norm_Pah = sum(P_Ah)
    P_Ah /= Norm_Pah

    diff[k] = abs(Norm_P-Norm_Ph)/Norm_P
    diffmax[k] = abs(np.max(P_A)-np.max(P_Ah))/np.max(P_A)
##
plt.figure(1)
plt.plot(NN, diff, 'o-')
plt.title('Percentual difference of Evidence', fontsize=14)
plt.xlabel('N', fontsize=14)
plt.ylabel(r'$\frac{|Z_m-Z_h|}{Z_m}$', fontsize=14)
plt.show()

plt.figure(2)
plt.plot(NN, diffmax, 'r-')
plt.title('Percentual difference of marginalized max', fontsize=14)
plt.xlabel('N', fontsize=14)
plt.ylabel(r'$\frac{|M_m-M_h|}{M_m}$', fontsize=14)
plt.show()

end=time.time()-start
print(f'time:{end}')