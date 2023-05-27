import time
import numpy as np
import scipy.stats as stat
import multiprocessing as mp
import matplotlib.pyplot as plt
import glob
import imageio.v2 as imageio

start=time.time()

def process_output(na, nb, nc, N, Na, Nb, pp, out_pro):
    '''
    Funzione che chima si serialmente la funzione per
    fattorizzare dei numeri, ma viene eseguita
    dai vari processi parallelamente

    Parameters
    ----------
    out_pro : method
        coda degli output
    '''
    out = {}
    out[pp] = post(na, nb, nc, N, Na, Nb, pp)

    out_pro.put(out)


def process_posterior(na, nb, nc, N, Na, Nb, PP, npro):
    '''
    Funzione che crea i processi che verranno utilizzati

    Parameters
    ----------
    ntf : list
        lista completa dei numeri da fattorizzare
    npro : int
        numero di processi da eseguire

    Returns
    ----------
    result : dict
        dizionario contenente tutti gli output
    '''

    out = mp.Queue() # coda degli output
    pro = [] #lista dei processi

    #ciclo sul numero di processi per crearli
    for i in range(npro):
        #definisco i processi
        p = mp.Process(target=process_output, args=(na, nb, nc, N, Na, Nb, PP[i], out))
        #li inserisco nella lista
        pro.append(p)
        #avvio i processi
        p.start()

    #racccolgo i risultati su un dizionario
    result = {}
    for i in range(npro):
        result.update(out.get())

    #attendo la fine dei processi
    for p in pro:
        p.join()

    return result


def run(na, nb, nc, N, Na, Nb, PP):

    npro = len(PP)
    result = process_posterior(na, nb, nc, N, Na, Nb, PP, npro)

    return result



def likelihood(x, p, n):
    '''
    likelihood,
    vedere docu di multivariate_hypergeom
    '''
    return stat.multinomial.pmf(x=x, n=n, p=p)

def prior(N):
    '''
    Prior
    '''
    return 2/((N+1)*(N+2)-2)

def post(na, nb, nc, N, Na, Nb, PP):

    P = np.zeros((N+1, N+1)) #Matrice che conterrà La postitior
    pa = Na*PP/N+(N-Na)*(1-PP)/(2*N)
    pb = Nb*PP/N+(N-Nb)*(1-PP)/(2*N)
    for i in range(N+1):
        for j in range(N+1):
            # vincolo di elezioni non truccate: votano solo N persone
            NC = N - Na[i] - Nb[j]
            pc = NC*PP/N+(N-NC)*(1-PP)/(2*N)
            if NC > 0:
                P[i, j] = likelihood(x=[na, nb, nc], p=[pa[i], pb[j], pc], n=na+nb+nc)*prior(N)

    return P

def GIF(path, name, ext='png'):

    path_in = path+'/*.'+ext
    path_out = path+f'/{name}.gif'

    imgs=[]

    file = glob.glob(path_in, recursive = True)
    file.sort(key=len)
    for im in file:
        imgs.append(imageio.imread(im))

    imageio.mimsave(path_out, imgs, duration = 500)


if __name__ == "__main__":

    N = 1000 # Popolazione totale avente diritto di voto

    Na = np.linspace(0, N, N+1, dtype=int) # possibili valori
    Nb = np.linspace(0, N, N+1, dtype=int) # che le variabili
    Nc = np.linspace(0, N, N+1, dtype=int) # possono assumere

    na = 20 # risultato
    nb = 10 # dei
    nc = 5 # sondaggi

    PP = np.linspace(0, 1, 50)

    P = run(na, nb, nc, N, Na, Nb, PP)

    pVA = np.zeros(len(PP))
    pVB = np.zeros(len(PP))
    H = {}
    pVAp = {}
    pVBp = {}
    Z = {}

    path = r'C:\Users\cakko\OneDrive\Documenti\Carlo\Unipi\Magistrale\TeoriaBayes\Elections\GIF'


    gridx, gridy = np.meshgrid(Na, Nb)
    for i, p, Pp in zip(range(len(PP)), P.values(), P.keys()):
        fig = plt.figure(i, figsize=(12,10))
        levels = np.linspace(0, np.max(p), 100)
        c=plt.contourf(gridx, gridy, p.T, levels=levels, cmap='gnuplot')
        plt.colorbar(c)
        plt.title(rf"P($N_A$,$N_B$|$n_A$={na},$n_B$={nb},$n_C$={nc}) with N={N} and p={Pp:.3f}", fontsize=16)
        plt.xlabel(r"$N_A$", fontsize=14)
        plt.ylabel(r"$N_B$", fontsize=14)
        plt.savefig(path + '/%d'%(i))
        plt.close(fig)

        for k in range(N+1):
            for j in range(N+1):
                if k>j and k>(N-j)/2:
                    pVA[i] = pVA[i] + p[k, j]
                if j>k and j>(N-k)/2:
                    pVB[i] = pVB[i] + p[k, j]

        #dato che il passo degli N è unitario basta sommare
        Norm_P = sum(sum(p)) #sommo due volte, per sommare su tutti gli indici
        p /= Norm_P
        #Marginalizzo su Na
        P_A = np.array([sum(p[i, :]) for i in range(N+1)])
        Norm_Pa = sum(P_A)
        P_A /= Norm_Pa
        #Marginalizzo su Nb
        P_B = np.array([sum(p[:, i]) for i in range(N+1)])
        Norm_Pb = sum(P_B)
        P_B /= Norm_Pb

        Z[Pp] = np.log(Norm_P)

        pVA[i] = pVA[i]/Norm_P
        pVB[i] = pVB[i]/Norm_P

        pVAp[Pp] = pVA[i]
        pVBp[Pp] = pVB[i]
        
        plt.figure(len(PP)+ 1)
        plt.title(f'Marginalized distributions P($N_A$|{na},{nb},{nc}) with N={N}', fontsize=14)
        plt.plot(Na, P_A, label=rf"p={Pp:.1f}")
        plt.xlabel(r"$N_A$", fontsize=14)
        plt.ylabel(r"P($N_A$ |$n_A$, $n_B$, $n_C$)", fontsize=14)
        plt.legend(ncol=2)
        plt.grid()
        
        plt.figure(len(PP)+ 2)
        plt.title(f'Marginalized distributions P($N_B$|{na},{nb},{nc}) with N={N}', fontsize=14)
        plt.plot(Nb, P_B, label=rf"p={Pp:.1f}")
        plt.xlabel(r"$N_B$", fontsize=14)
        plt.ylabel(r"P($N_B$ |$n_A$, $n_B$, $n_C$)", fontsize=14)
        plt.legend(ncol=2)
        plt.grid()

        post = p.ravel()
        h = 0

        for pi in post:
            if pi<1e-13:
                pass
            else:
                h = h + pi*np.log(pi)
        #compute information 
        H[Pp] = -h



    plt.figure(2*len(PP)+1)
    plt.title(f'Probability of victory as a function of p', fontsize=14)
    plt.plot(pVAp.keys(), pVAp.values(), 'ro', label=rf"P($V_A$)")
    plt.plot(pVBp.keys(), pVBp.values(), 'go', label=rf"P($V_B$)")
    plt.xlabel(r"$p$", fontsize=14)
    plt.ylabel(r"P($V_X$)", fontsize=14)
    plt.legend()
    plt.grid()

    p_temp = np.array([c for c in Z.keys()])
    z_temp = np.array([c for c in Z.values()])
    h_temp = np.array([c for c in H.values()])
    p_plot = np.array([])
    z_plot = np.array([])
    h_plot = np.array([])

    for i in range(len(p_temp)):
        min_p_temp  = np.min(p_temp)
        index_min_p = np.where(p_temp == min_p_temp)[0][0]
        min_z_temp  = z_temp[index_min_p]
        index_min_z = np.where(z_temp == min_z_temp)[0][0]
        min_h_temp  = h_temp[index_min_p]
        index_min_h = np.where(h_temp == min_h_temp)[0][0]

        p_plot = np.insert(p_plot, len(p_plot), min_p_temp)
        z_plot = np.insert(z_plot, len(z_plot), min_z_temp)
        h_plot = np.insert(h_plot, len(h_plot), min_h_temp)


        p_temp = np.delete(p_temp, index_min_p)
        z_temp = np.delete(z_temp, index_min_z)
        h_temp = np.delete(h_temp, index_min_h)

    dZ = np.sqrt(h_plot/N)

    plt.figure(2*len(PP)+2)
    plt.title(f'Evidence as function of p', fontsize=16)
    #plt.plot(Z.keys(), Z.values(), 'k-')
    plt.errorbar(p_plot, z_plot, dZ,  ecolor='k', fmt='.')
    plt.xlabel(r"$p$", fontsize=14)
    plt.ylabel(r"logZ", fontsize=14)
    plt.grid()



    end=time.time()-start
    print(f'time:{end}')
    GIF(path, 'Posterior')

    plt.show()


