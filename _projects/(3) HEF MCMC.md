---
name: MCMC for multi-particle phase space 
tools: [MCMC, QFT, Python, Radboud University]
image: /assets/images/mcmc.png
description: Python implementation of a MCMC multi-particle phase space sampler for massless particles.
---

## Introduction

For the master's course Mone Carlo Techniques in 2017, I assisted professors Kleiss' research by implementing RAMBO in Python to uniformly sample $$n$$-particle phase space and 
 minimize the following quantity:

$$R = - \sum_{ab} Q_a Q_b (1 - \cos(\theta_{ab})) \quad \text{1.)}$$

Where $$Q_{a,b}$$ are fermion charches and 

$$\cos\left(\theta_{ij}\right) = 1-\frac{\vec{p_i} \vec{p_j}}{p^0_{i} p^0_{j}}$$ 

is the angle between two fermions. This quantity $$R$$ resembles the *eikonal factors* of a QFT matrix element.
The case where $$R\to 0$$ is a relevant limiting case. The details of this research are explained in [1], but a short summary will be presented here.

Quantum Chromodynamics (QCD) raditation is dominant in Large Hadron Collider (LHC) experiments. However, other types of radiation are allowed in the Standard Model.
This radiation should be interleaved with the dominant QCD radiation in order to get a precise understanding of the LHC experiments.

![](/assets/images/project_mcmc/cms.png)
***Figure 1.)*** *Simulating the outcome of LHS collisions is a major field of research, for which powerful numerical methods have to be developed [2]*

While most event generators (which are used to simulate the physics in LHC experiments) can simulate Quantum Electrodynamics (QED) radiation, 
these simulations lack some key physical properties. Devising an algorithm to include the correct physics is non-trivial.
The authors of [1] introduce a parton shower formalism which produces coherent QED radiation and which can be interleaved in a regular QCD shower. 
For this algorithm, they estimate an upper bound of the splitting function (function that gives the probability of a particle of type A emitting a particle of type B) of a certain physical process.
This upper bound correlates with the quantity R.

Luckily, I did not have to bother with any of that. My goal for this project was simply to sample the $$n$$-particle phase space using an algorithm
called RAMBO, and then investigate if simulated annealing could be used to to minimize $$R$$.

## RAMBO

In order to calculate branching ratios or decay rates in Quantum Field Theory (QFT), we have to evaluate integrals over
a Lorentz invariant measure. This measure ensures that the quantities we calculate are independent of our frame of reference,
which is crucial when one is doing relativistic physics. For a single particle with mass $m$, the phase space is given by

$$d\Pi^{(1)} = \frac{d^4 p}{(2\pi)^4} (2\pi) \delta(p^2 - m^2) \Theta(E) $$

Where we apply the usual convention that $$\hbar=c=1$$. $$\Theta$$ is the step function which ensures that
$$E>0$$ and the delta function ensures that the particle is on-shell (follows the relativistic mass energy relation $$E^2 - p^2 = m^2$$).

We can extend this definition to $$n$$ massless particle phase space:

$$ d\Pi^{(n)} = \prod^n_{i=1}\left [ \frac{d^4 p}{(2\pi)^4} (2\pi) \delta(p^2) \Theta(E) \right ] \left(P - \sum_{i=1}^n p_i\right) $$

where we now also account for the conservation of momentum by introducing an additonal delta function. Sampling this phase space 
uniformly is non-trivial, but possible with the RAMBO algorithm [4]. This algorithm allows one to sample isotropically from a massless 
$n$-particle phaes space for a given invariant mass $S_{\text{inv}} = \sum_i^n (p_i)^2$. The Python code I used at the time
looks as follows:

```python
import numpy as np


def k_dot_q(k, q):
    '''
    Minkowski product

    :param k: 4-vector
    :param q: 4-vector
    :return: innerproduct using metric diag(1,-1,-1,-1)
    '''
    return (k[0] * q[0] - np.dot(k[1:4], q[1:4]))


def RAMBO(n, s):
    '''
    RAMBO (RAndom Momenta BOoster)

    A DEMOCRATIC MULTI-PARTICLE PHASE SPACE GENERATOR
    AUTHORS:  S.D. ELLIS,  R. KLEISS,  W.J. STIRLING
    Python version by Roeland Wiersema

    The RAMBO algorithm aims at generating points in an n-particle phase space
    in the ultra-relativistic limit, where the particle masses can be neglected.
    The task is not only to obtain four-momenta that satisfy the constraints of
    energy and momentum conservation, but also we have to ensure that the whole
    phase space is covered, with as uniform a density as possible.

    :param n: Number of particles
    :param s: total invariant mass: sqrt(s)
    :return p: n momentum fourvectors in (n,4) numpy array
    '''
    q = np.zeros((n, 4), dtype='float64')
    for i in range(n):
        q[i][0] = -np.log(np.random.rand() * np.random.rand())
        C = 2 * np.random.rand() - 1
        S = np.sqrt(1. - C * C)
        F = 2 * np.pi * np.random.rand()
        q[i][1] = q[i][0] * S * np.sin(F)
        q[i][2] = q[i][0] * S * np.cos(F)
        q[i][3] = q[i][0] * C
    # get k
    k = np.sum(q, axis=0)
    # calculate u using metric g = diag(1,-1,-1,-1)
    u = np.sqrt(k_dot_q(k, k))
    # initialize
    p = np.empty((n, 4), dtype='float64')

    for i in range(n):
        # assign p0
        p[i][0] = (np.sqrt(s) / u ** 2) * (k_dot_q(k, q[i]))
        # assign p_vector
        p[i][1:4] = (np.sqrt(s) / u ** 2) * \
                    (u * q[i][1:4] - k[1:4] * (k_dot_q(k, q[i]) + u * q[i][0]) / (u + k[0]))

    return p
```
Whi

## Simulated Annealing

We define a Gibbs distribution

$$p(\theta_1,...,\theta_n,q_1,...q_n) = \frac{1}{Z} \exp(-\beta R)$$

where the energy $$R$$ is defined as in equation 1.). To perform a random walk we vary the angles $\theta_i$ and flip the charges $q$.
To shift the angles between the momentum four vectors we have to ensure that the invariant mass $S_{\text{inv}}$ stays the same. 
We also require the momenta to remain massless, so $p^{\mu} p_{\mu}=0$. 
Finally, we need the vector parts of the four-momentum to add up to zero, so $\sum_{i=1}^{n}\vec{p_i}=0$ so that we stay in the center of
mas system. To achieve all of this, we perform a Lorentz boost to the rest system of two randomly selected fermions, and use a rotation
in the $$x$$, $y$ or $z$ direction to rotate the two chose four vectors. Then we boost back to the initial system. We define

$$
\begin{align}
p &= p_a+p_b\nonumber\\
s &= p^2=q^2\nonumber\\
\text{where } q &= (\sqrt{s},0,0,0)\nonumber\\
\text{and } q &= q_a+q_b\nonumber
\end{align}
$$

Then, to boost the frame $S$ to the frame $$S^\prime$$ with momentum we use the following transformation

$$
\Lambda(p\rightarrow q)^{\mu}_{\nu} = \delta^{\mu}_{\nu} - \frac{1}{s+p^0\sqrt{s}}(p+q)^{\mu}(p+q)_{\nu}+\frac{2}{s}q^{\mu}p_{\nu}
$$

which gives
 
$$
\begin{align}
q^{0}_{a} &= \Lambda(p\rightarrow q)^{0}_{\nu} p^{\nu}_{a} = \frac{p \cdot p_{a}}{\sqrt{s}}\\
q^{i}_{a} &= \Lambda(p\rightarrow q)^{i}_{\nu} p^{\nu}_{a} = p^{i}_{a} - \frac{p^{0}_{a}+q^{0}_{a}}{\sqrt{s}+p^0} p^i
\end{align}
$$

for the boost **to** the rest system. In this rest system we perform a rotation $q^\prime = R^\alpha(q, \delta \theta)$, where $R^\alpha$ is the standard
3D rotation around the $\alpha=x,y,z$ axis. Then we boost **from** the rest system back to the original frame of reference:

$$
\begin{align}
p^{0,\prime}_{a} &= \Lambda(q^\prime\rightarrow p^\prime)^{0}_{\nu} q^{\nu,\prime}_{a} = \frac{p^0 q^0_a+p^i q^i_a}{\sqrt{s}}\\
p^{i,\prime}_{a} &= \Lambda(q^\prime\rightarrow p^\prime)^{i}_{\nu} q^{\nu, \prime}_{a} = q^{i}_{a} + \frac{q^{0}_{a}+p^{0,\prime}_{a} }{\sqrt{s}+p^0} p^i
\end{align}
$$

Additionally, we also swap the fermion charges $Q_a$ and $Q_b$ randomly, so that charge is conserved.

Next, we perform the standard simulated annealing schedule, where we start from some low initial temperature $\beta_0$ in our Gibbs distribution
and slowly anneal to a high value of $\beta$ to freeze the dynamics of the random walk. Thermal fluctuations at higher temperature
allow us to escape local minimia, and if we cool slowly enough, we expect to find the global minimimum of the energy function.

In the plot below, we see that this method was succesful in finding a $R\approx 0$ momentum state

![](/assets/images/project_mcmc/result.png)
***Figure 2.)*** *Simulated annealing results for 6 fermions with an invariant mass of $S_{inv} = 20$MeV. The efficiency leaves something to be desired, but simulated annealing seems to work for sampling 
$R \approx 0$* states.

## Bibliography
[1] *Final-state QED multipole radiation in antenna parton showers*, **Journal of High Energy Physics**, R. Kleiss and  R. Verheyen, 2017 

[2] *CMS Collaboration*, **CERN**, http://cds.cern.ch/record/1309898, 2010 

[3] *Lecture notes Quarks and Leptons*, **Frank Krauss**, https://www.ippp.dur.ac.uk/~krauss/Lectures/QuarksLeptons/, 2005

[4]  *A New Monte Carlo Treatment of Multiparticle Phase Space at High-energies*, **R. Kleiss, W.James Stirling, S.D. Ellis**, Computer Physics Communications, 1985

    Comput.Phys.Commun. 40 (1986) 359