"""
Simple code that implements the nested sampling
to compute the evidence of a multidimensional gaussian
"""
import time
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as interp


def log_likelihood(ni, Ni, D, N, bound):
    """
    log likelihood of gaussian

    Parameters
    ----------
    x : array or matrix
        array of parameters
    D : int
        dimension of parameter space

    Return
    ------
    likelihood : list or float
        log likelihood
    """

    NC = N - sum(Ni[:D-1]) # vincolo sulla popolazione
    new_Ni = np.copy(Ni)   # copio l'array per evitare problemi di sovrascrittura
    new_Ni[-1] = NC        # valore per calcolo likelihood che rispetta il vincolo

    int_Ni = np.array([], dtype=int)
    for nn in new_Ni:      # prendo la parte intera
        int_Ni = np.insert(int_Ni, len(int_Ni), int(nn))

    X = np.zeros((3, 2), dtype=int)
    for i, b_min, b_max in zip(range(D), bound[0::2], bound[1::2]):
        if i == 2: break

        if int_Ni[i] == b_max:
            X[:,i] = np.array([int_Ni[i]-2, int_Ni[i]-1, int_Ni[i]])
        elif int_Ni[i] == b_min:
            X[:,i] = np.array([int_Ni[i], int_Ni[i]+1, int_Ni[i]+2])
        elif b_min < int_Ni[i] < b_max :
            X[:,i] = np.array([int_Ni[i] - 1, int_Ni[i], int_Ni[i] + 1])

    LL = np.zeros((3, 3))
    x = X[:,0]
    y = X[:,1]

    # se non rispetta il vincolo, probabilità zero
    if NC < 0:
        likelihood = - 30 #1e-70 è abbastanza zero
    else:
        for i in range(len(x)):
            for j in range(len(y)):
                LL[i,j] = stat.multivariate_hypergeom.pmf(x=ni, m=[x[i], y[j], N-x[i]-y[j]], n=sum(ni))

        Lf = interp((x, y), LL)

        L = Lf(np.array([Ni[0], Ni[1]]))
        if L!=L:
            L = 1e-30
        if L < 1e-30: #per vevitare problemi nel log
            likelihood = - 30
        else:
            likelihood =  np.log(L)
    #print()
    return likelihood


def samplig(ni, x, D, bound, step, N):
    """
    Sampling a new point of parameter space from
    a uniform distribution as proposal with the
    constraint to go up in likelihood

    Parameters
    ----------
    x : 1darray
        values of parameters and the relative likelihood
        x[:len(x)-1] = parameters
        x[len(x)-1] = likelihood(parameters)
    D : int
        dimension for parameter space, len(x) = D+1
    bound: float
        bounds of parameter space

    Return
    ------
    new_sample : 1darray
        new array replacing x with higher likelihood
    accept : int
        number of moves that have been accepted
    reject : int
        number of moves that have been rejected
    """

    logLmin = x[D] #worst likelihood
    point = x[:D]  #point in the parameter space
    #step = 0.1     #initial step of the algorithm, to set
    accept = 0     #number of accepted moves
    reject = 0     #number of rejected moves
    trial  = 0     #numero tentativi fatti
    step = step

    while True:
        #array initialization
        new_sample = np.zeros(D+1)
        #loop over the components
        for i in range(D):
            #we sample a trial variable
            #new_sample[:D][i] = point[i] + np.random.uniform(-step, step)
            new_sample[i] = np.random.normal(point[i], step[i])
            #if it is out of bound...
            #while np.abs(new_sample[:D][i]) > bound:
            while new_sample[i] <= bound[2*i] or new_sample[i] >= bound[2*i + 1]:
                #...we resample the variable
                #new_sample[:D][i] = point[i] + np.random.uniform(-step, step)
                new_sample[i] = np.random.normal(point[i], step[i])
        #computation of the likelihood associated to the new point
        new_sample[D] = log_likelihood(ni, new_sample[:D], D, N, bound)

        #if the likelihood is smaller than before we reject
        if new_sample[D] < logLmin:
            reject += 1
        #if greater we accept
        if new_sample[D] > logLmin:
            trial += 1
            accept += 1
            point[:D] = new_sample[:D]

            if trial > 1:#ACHTUNG
                """
                the samples must be independent. We trust
                that they are after 40 accepted attempts,
                but we should compute the correlation of the D
                chains and the autocorrelation time in order
                to know what to write instead of 40, which is
                computationally expensive
                """
                break

        # We change the step to go towards a 50% acceptance
        if accept != 0 and reject != 0:
            if accept > reject :
                step *= np.exp(1.0 / accept);
            if accept < reject :
                step /= np.exp(1.0 / reject);

    return new_sample, accept, reject


def nested_samplig(ni, N, M, D, bound, tau=1e-6, verbose=False):
    """
    Compute evidence, information and distribution of parameters

    Parameters
    ----------
    N : int
        number of points
    D : int
         dimension for parameter space
    bound: float
        bounds of parameter space
    tau : float
        tollerance, the run stops when the
        variation of evidence is smaller than tau
    verbose : bool, optional
        if True some information are printed during
        the execution to see what is happening

    Retunr
    ------
    calc : dict
        a dictionary which contains several information:
        "evidence"        : logZ,
        "error_lZ"        : error,
        "posterior"       : grid[:, :D],
        "likelihood"      : grid[:,  D],
        "prior"           : prior_sample,
        "prior_mass"      : prior_mass,
        "number_acc"      : accepted,
        "number_rej"      : rejected,
        "number_steps"    : iter,
        "log_information" : logH_list,
        "list_evidence"   : logZ_list

    """

    grid = np.zeros((M, D + 1))

    prior_mass = [] #integration variable
    logH_list  = [] #we keep the information
    logZ_list  = [] #we keep the evidence

    logH = -np.inf  # ln(Information, initially 0)
    logZ = -np.inf  # ln(Evidence Z, initially 0)

    #indifference principle, the parameters' priors are uniform
    prior_sample = np.zeros((M, D))
    for i in range(D):
        prior_sample[:, i] = np.random.uniform(bound[2*i], bound[2*i + 1], size=M)

    #initialization of the parameters' values
    grid[:, :D] = prior_sample
    #likelihood initialization
    for i in range(M):
        grid[i, D] = log_likelihood(ni, prior_sample[i,:], D, N, bound)
    #print(log_likelihood(ni, prior_sample[i,:], D, N))
    #quit()
    # Outermost interval of prior mass
    logwidth = np.log(1.0 - np.exp(-1.0/M))

    iter     = 0   #number of steps
    rejected = 0   #total rejected steps
    accepted = 0   #total accepted steps

    while True:
        iter += 1                        #we refresh the number of steps
        prior_mass.append(logwidth)      #we keep the integration variable

        Lw_idx = np.argmin(grid[:, D])   #index for the parameters with the worst likelihood, i.e. the smallest one
        logLw = grid[Lw_idx, D]          #value of the worst likelihood
        #print(logLw, logwidth)
        #quit()
        #np.logaddexp(x, y) = np.log(np.exp(x) + np.exp(y))
        logZnew = np.logaddexp(logZ, logwidth+logLw)
        #print(logZ, logZnew)
        #quit()

        logZ = logZnew           #we refresh the evidence
        logZ_list.append(logZ)   #we keep the value of the evidence

        #we compute the information and keep it
        logH = np.logaddexp(logH, logwidth + logLw - logZ + np.log(logLw - logZ))
        logH_list.append(logH)

        #new sample used to replace the points we have to delete
        sampling_step = np.array([np.std(grid[:, jj]) for jj in range(D)])
        new_sample, acc, rej = samplig(ni, grid[Lw_idx], D, bound, sampling_step, N)
        accepted += acc #we refresh the total accepted steps
        rejected += rej #we refresh the total rejected steps

        grid[Lw_idx] = new_sample #replacement
        logwidth -= 1.0/M       #interval shrinking

        if verbose :
            #evidence error computed each time for the print
            error = np.sqrt(np.exp(logH)/M)
            print(f"Iter = {iter} acceptance = {accepted/(accepted+rejected):.3f} logZ = {logZ:.3f} error_logZ = {error:.3f} H = {np.exp(logH):.3f} \r", end="")

        if iter > 3:
            #break
            if abs((logZ_list[-1] - logZ_list[-2])/logZ_list[-2]) < tau :
                break

    #evidence error
    error = np.sqrt(np.exp(logH)/M)

    calc = {
            "evidence"        : logZ,
            "error_lZ"        : error,
            "posterior"       : grid[:, :D],
            "likelihood"      : grid[:,  D],
            "prior"           : prior_sample,
            "prior_mass"      : np.array(prior_mass),
            "number_acc"      : accepted,
            "number_rej"      : rejected,
            "number_steps"    : iter,
            "log_information" : np.array(logH_list),
            "list_evidence"   : np.array(logZ_list)
    }

    return calc


def plot_hist_par(prior, posterior, D, save=False, show=False):
    """
    Plot of posterior vs prior for all parameters

    Parameters
    ----------
    prior : 2darray
        matrix which contains the prior of all parameters
    posterior : 2darray
        matrix which contains the posterior of all parameters
    D : int
         dimension of the parameter space
    save : bool, optional
        if True all plots are saved in the current directory
    show : bool, optional
        if True all plots are showed on the screen
    """

    for pr, ps, k in zip(prior.T, posterior.T, range(D)):

        fig = plt.figure(k+1)
        plt.title(f"Confronto per il {k+1}esimo parametro")
        plt.xlabel("bound")
        plt.ylabel("Probability density")
        plt.hist(pr, bins=int(np.sqrt(N-1)), density=True, histtype='step', color='blue', label='prior')
        plt.hist(ps, bins=int(np.sqrt(N-1)), density=True, histtype='step', color='black', label="posterior")
        plt.legend(loc='best')
        plt.grid()

        if save :
            plt.savefig(f"parametro{k+1}")
            plt.close(fig)



    fig = plt.figure(2*k)
    plt.title(f"Confronto per i parametri")
    plt.xlabel("bound")
    plt.ylabel("Probability density")
    plt.hist(prior.T[0], bins=int(np.sqrt(N-1)), density=True, histtype='step', color='black', label='prior')
    plt.hist(posterior.T[0], bins=int(np.sqrt(N-1)), density=True, histtype='step', color='red', label=r"post $N_A$")
    plt.hist(posterior.T[1], bins=int(np.sqrt(N-1)), density=True, histtype='step', color='green', label=r"post $N_B$")
    plt.legend(loc='best')
    plt.grid()

    if show :
        plt.show()


if __name__ == "__main__":

    np.random.seed(69420)
    #number of points
    N = int(1e3)
    M = int(2e4)

    bound = np.array([200, 650, 200, 650, 0, N], dtype=int)
    #dimension
    D = len(bound)//2
    ni = np.array([20, 10, 5]) #sondaggio

    start = time.time()

    NS = nested_samplig(ni, N, M, D, bound, verbose=True)

    evidence        = NS["evidence"]
    error_evidence  = NS["error_lZ"]
    posterior_param = NS["posterior"]
    likelihood      = NS["likelihood"]
    prior_param     = NS["prior"]
    prior_mass      = NS["prior_mass"]
    acc             = NS["number_acc"]
    rej             = NS["number_rej"]
    number_iter     = NS["number_steps"]
    log_information = NS["log_information"]
    list_evidence   = NS["list_evidence"]

    print(f"Evidence sampling    = {evidence:.3f} +- {error_evidence:.3f}")
    #print(f"Theoretical evidence = {-D*np.log(2*bound):.3f}")

    print(f"Number of iterations = {number_iter}")

    acceptance = acc/(acc+rej)
    print(f"Acceptance = {acceptance:.3f}")

    end = time.time() - start

    print(f"Elapsed time = {end//60:.0f} min and {end%60:.0f} s")

    plot_hist_par(prior_param, posterior_param, D, show=True)
