############ Mini-projet Ma322/ Emmanuel ODENYA VD1 ############

##### Imports #####

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

##### My work #####


########### EQUATION INTEGRALE ##########

## Présentation de Benchmark utilisée

N = 100
a = -1
b = 1
K = 2

### Question 2.a - Script Mat

def f(x):
    return np.cos(np.pi*x/2) - 8/np.pi

def u(x):
    return np.cos(np.pi*x/2)

X = np.linspace(a, b, N+1)
h = (b-a)/N
I = np.eye(N+1)


def constructMatA(K):
    '''Je construis la matrice A telle que je la définis dans mon rapport'''
    
    A = K*np.ones((N+1, N+1))
    for i in range(len(X)):
        for j in range(1, N):
            A[i, j] = 2*A[i,j]
    return A

A = constructMatA(K)

def Mat(f, A):
    M = np.zeros((N+1, N+1))
    F = np.zeros(N+1)
    for i in range(len(X)):
        M[i, :] = I[i, : ] - (h/2)*A[i, :]
        F[i] = f(X[i])
    return F, M

## Question 2.b - Test numérique
      
Res = Mat(f, A)
V = np.linalg.solve(Res[1], Res[0]) # On résout le système MU = F

# Solution exacte connaissant u(x)

U = np.zeros(N+1)
for i in range(len(X)):
    U[i] = u(X[i])

## Question 2.c - Tracé des solutions

plt.plot(X, U, label = 'Solution exacte', color = 'blue')
plt.plot(X, V, 'x', label = 'Solution approchée', color = 'red')
plt.grid()
plt.title('Equation intégrale résolue')
plt.legend()
plt.show()


## Question 2.d - Erreur calculée : || U - V ||

err = np.linalg.norm(U - V)

print('\n Erreur entre solution approchée et solution exacte vaut \n', 'err =', err, '\n')


## Equation de Love en électrostatique
        
def funcK(x,t):
    '''Ici K n'est plus une constante mais une fonction de 2 variables'''
    ans = 1 + (x-t)**2
    return 1/(np.pi*ans)


t = np.linspace(a, b, N+1)


MatK = np.zeros((len(t), len(t)))

for i in range(len(t)):
    for j in range(len(t)):
        MatK[i, j] = funcK(X[i], t[j])


def foncf(x):
    return 1 # Dans le cas de l'équation de Love, la fonction f est constante


InterA = np.zeros((N+1, N+1))
'''Je crée une matrice intermédiaire de 0 et ayant les 1ère et dernières colonnes de K sur ses 1ère et dernière colonnes'''
'''On aurait bien pu utiliser une fonction comme précédemment en prenant K comme une fonction en entrée et pas une constante'''
InterA[:, 0] = MatK[:, 0]
InterA[:, N] = MatK[:, N]

NewA = 2*MatK - InterA

Res1 = Mat(foncf, NewA) # On calcule la solution numérique en utilisant le script Mat appliquée aux nouveaux paramètres
U_num = np.linalg.solve(Res1[1], Res1[0])

print('La solution numérique pour cette équation de Love est \n ', 'U \n', U_num)

# Courbe représentative de u

plt.plot(X, U_num)
plt.title('\n Courbe de u pour Love en électrostatique')
plt.grid()
plt.show()


########## CIRCUIT RLC #########

# On définit les paramètres du problème

R_rlc = 3
L_rlc = 0.5
C = 1e-6
e = 10
T_rlc = np.linspace(0, 2, 1000)
N_rlc = T_rlc.size


### Question 3 - Fonction rlcprim

def rlcprim(Y, t):
    Yp1 = Y[1]/C
    Yp2 = (e - Y[0] - R_rlc*Y[1])/L_rlc
    return np.array([Yp1, Yp2])
    

### Question 4 - Runge-Kutta 4

# On programme ici la méthode de Runge Kutta d'ordre 4 à partir du schéma de la méthode

def RungeKutta(f, h, y0):
    c = y0.size
    Yrk = np.zeros((N_rlc,c))
    Yrk[0, : ] = y0
    for i in range(N_rlc-1):
        k_1 = f(Yrk[i , : ], 0)
        k_2 = f(Yrk[i , : ] + (h/2)*k_1, 0)
        k_3 = f(Yrk[i , : ] + (h/2)*k_2, 0)
        k_4 = f(Yrk[i , : ] + h*k_3, 0)
        Yrk[i+1, : ] = Yrk[i , : ] + (h/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
    return Yrk


## Résolution du RLC

Y0_rlc = np.array([0, 0])
RK_rlc = RungeKutta(rlcprim, 0.002, Y0_rlc)

rlc_odeint = odeint(rlcprim, Y0_rlc, T_rlc)

## Tension s

plt.plot(T_rlc, rlc_odeint[:,0], label = 'Tension odeint')
plt.plot(T_rlc, RK_rlc[:,0], label = 'Tension RK')
plt.legend()
plt.xlabel('t')
plt.ylabel('s')
plt.title('Tension du circuit RLC')
plt.grid()
plt.show()

## Intensité i

plt.plot(T_rlc, rlc_odeint[:,1], label = 'Intensité odeint')
plt.plot(T_rlc, RK_rlc[:,1], label = 'Intensité RK')
plt.legend()
plt.xlabel('t')
plt.ylabel('i')
plt.title('Intensité du courant du RLC')
plt.grid()
plt.show()


## En modifiant les paramètres du problème, notamment C (on essaie d'approcher ainsi un problème réel)

C_test = 1000*1e-6 # On réalise ce test en utilisant une autre valeur de C

def rlcprim_test(Y, t):
    Yp1 = Y[1]/C_test
    Yp2 = (e - Y[0] - R_rlc*Y[1])/L_rlc
    return np.array([Yp1, Yp2])

rlc_odeint_test = odeint(rlcprim_test, Y0_rlc, T_rlc)
RK_rlc_test = RungeKutta(rlcprim_test, 0.002, Y0_rlc)

plt.plot(T_rlc, rlc_odeint_test[:,0], label = 'Tension 2 odeint')
plt.plot(T_rlc, RK_rlc_test[:,0], label = 'Tension 2 RK')
plt.legend()
plt.xlabel('t')
plt.ylabel('s')
plt.title('Tension du RLC pour C = 1e-3')
plt.grid()
plt.show()

plt.plot(T_rlc, rlc_odeint_test[:,1], label = 'Intensité 2 odeint')
plt.plot(T_rlc, RK_rlc_test[:,1], label = 'Intensité 2 RK')
plt.legend()
plt.xlabel('t')
plt.ylabel('i')
plt.title('Intensité du RLC pour C = 1e-3')
plt.grid()
plt.show()

# L'affichage des résultats nous permet de réaliser que dans ce cas-ci, les solutions par odeint et RK4 se superposent



######### MOTEUR A COURANT CONTINU #############

T_mcc = np.linspace(0, 80, 1000) #On discrétise t avec un pas h = 0,08

# On entre ici tous les paramètres électromécaniques du problème

R_mcc = 5
L_mcc = 0.05
Ke = 0.2
Kc = 0.1
Fm = 0.01
Jm = 0.05


def u_MCC(t): # On définit la tension appliquée au moteur telle que définie dans le problème
    if 10 <= t <= 50 :
        return 5
    else :
        return 0

    
### Question 3 - Fonction moteurCC

def moteurCC(Y,t):
    Yp1 = (u_MCC(t) - R_mcc*Y[0] - Ke*Y[1])/L_mcc
    Yp2 = (Kc*Y[0] - Fm*Y[1])/Jm
    return np.array([Yp1, Yp2])


### Question 4 - Résolution odeint

Y0_mcc = np.array([0,0])
mcc_odeint = odeint(moteurCC, Y0_mcc, T_mcc)

# plt.plot(T_mcc, mcc_odeint[:,0], label = 'Intensité i')
# plt.legend()
# plt.grid()
# plt.show()
''' Permet de remarquer que l'intensité a la même allure que le couple moteur'''

## Vitesse angulaire

plt.plot(T_mcc, mcc_odeint[:,1])
plt.xlabel('temps t')
plt.ylabel('Vitesse angulaire w')
plt.title('Evolution de la vitesse angulaire (MCC)')
plt.grid()
plt.show()

## Evolution du couple moteur

I = mcc_odeint[:,0] # On extrait les valeurs calculées de l'intensité de la table que retourne le solveur odeint

Cm = Kc*I # Le couple moteur est défini proportionnellement à l'intensité

plt.plot(T_mcc, I)
plt.xlabel('temps t')
plt.ylabel('Couple moteur $C_m$')
plt.title('Evolution du couple moteur (MCC)')
plt.grid()
plt.show()



########## MOUVEMENT D'UNE FUSEE ###############

### Question 1 - Fonction Fusée

def fusee(Y, t):
    # On commence par définir les paramètres de la fusée
    D = 4
    a = 8*10**3
    g = 9.81
    k = 0.1
    u = 2*10**3
    
    Yprime = np.zeros(3)
    if (Y[1] < 80) :
        Y[1] = 80 # On fixe la masse de la fusée à la fin de la combustion à t = 80
        D = 0 # La combustion s'arrêtant à t = 80, le débit de gaz est nul à partir de ce moment
        
    #On peut alors définir les 3 coefficients de F(t,Y) à partir de leurs expressions
    Yprime[0] = D*u/Y[1] - g - k*np.exp(-Y[2]/a) * ((Y[0])**2)/Y[1]
    Yprime[1] = -D
    Yprime[2] = Y[0]
    return Yprime


### Question 2 - Odeint

T_fusee = np.linspace(0, 160, 100) # On discrétise l'intervalle de temps avec un pas h = 1.6, suffisant pour observer tous les résultats
# Remarquons que pour un pas h trop faible, l'algorithme rencontre des problème de calcul


Y0_fusee = np.array([0, 400, 0])
fusee_odeint = odeint(fusee, Y0_fusee, T_fusee)

plt.plot(T_fusee, fusee_odeint[:,0])
plt.title('Evolution de la vitesse')
plt.xlabel('temps t')
plt.ylabel('Vitesse v')
plt.grid()
plt.show()

# # On peut aussi représenter l'évolution de la masse au cours du mouvement de la fusée (ce qui n'est pas demandé ici)
# plt.plot(T_fusee, fusee_odeint[:,1], label = 'Evolution de la masse')
# plt.legend()
# plt.grid()
# plt.show()

plt.plot(T_fusee, fusee_odeint[:,2])
plt.xlim(0, 80) # On limite les abscisses pour n'afficher que l'évolution de l'altitude pendant la phase de propulsion qui dure 80s
plt.ylim(0, 31000)
plt.title('Trajectoire dans la phase de propulsion')
plt.xlabel('temps t')
plt.ylabel('Altitude z')
plt.grid()
plt.show()



########## MODELE PROIE-PREDATEUR ############

# On définit les paramètres du problème

alpha1 = 3
beta1 = 1
alpha2 = 2
beta2 = 1

# On définit les conditions initiales

y1_0 = 5
y2_0 = 3

T_proie = np.linspace(0, 10, 1000)
N_proie = T_proie.size


### Fonction proies-prédateur

# On code la fonction Y' = F(t, Y) telle que définie à partir du système d'équations différentielles

def ProiePredateur(Y, t):
    Yprime = np.zeros(2)
    Yprime[0] = alpha1*Y[0] - beta1*Y[0]*Y[1]
    Yprime[1] = -alpha2*Y[1] + beta2*Y[0]*Y[1]
    return Yprime


### Question 1 - Proies sans prédateur


Y0_proie = np.array([5, 0]) # On fixe la population de prédateurs à 0 à l'instant initial

Proie = odeint(ProiePredateur, Y0_proie, T_proie)

plt.plot(T_proie, Proie[:,0], color = 'green')
plt.title('Evolution des proies en absence de prédateurs')
plt.grid()
plt.show()


### Question 2 - Prédateurs sans proies
    
Y0_predateur = np.array([0, 3]) ## On fixe la poulation de proies à 0 à l'instant initial
Predateur = odeint(ProiePredateur, Y0_predateur, T_proie)


plt.plot(T_proie, Predateur[:,1], color = 'red')
plt.title('Evolution des prédateurs en absence de proies')
plt.grid()
plt.show()


### Question 3 - Euler explicite

# On programme la méthode d'Euler explicite à partir du schéma de la méthode

def EulerExplicite(f, h, y0):
    p = y0.size
    Ye = np.zeros((N_proie, p))
    Ye[0, : ] = y0
    for i in range(N_proie-1): 
        Ye[i+1, : ] = Ye[i , : ] + h*f(Ye[i , : ], 0)
    return Ye

Y0 = np.array([y1_0, y2_0])

Euler = EulerExplicite(ProiePredateur, 0.01, Y0) # On résout le système pour trouver y1 et y2

plt.plot(T_proie, Euler[:,0], label = 'Evolution proies', color='cyan')
plt.plot(T_proie, Euler[:,1], label = 'Evolution prédateur', color='magenta')
plt.legend()
plt.title('Modèle proie-prédateur avec Euler explicite')
plt.grid()
plt.show()


### Question 4.a - Odeint

ProiePredateur_odeint = odeint(ProiePredateur, Y0, T_proie)

plt.plot(T_proie, ProiePredateur_odeint[:,0], label = 'Evolution des proies', color='green')
plt.plot(T_proie, ProiePredateur_odeint[:,1], label = 'Evolution des prédateurs', color='red')
plt.legend()
plt.title('Modèle proie-prédateur avec odeint')
plt.grid()
plt.show()


### Question 5.a - Portrait de phase

plt.plot(Euler[:,0], Euler[:,1], label = 'Euler explicite', color='orange')
plt.plot(ProiePredateur_odeint[:,0], ProiePredateur_odeint[:,1], label = 'odeint',  color='blue')
plt.title('Portrait de phase de la solution')
plt.legend()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.grid()
plt.show()



### Question 6. Tests en faisant varier les paramètres du problème

# Je reprends ici la fonction ProiePredateur en ajoutant les paramètres comme arguments en entrée
# pour pouvoir les modifier au moment de l'appel de la fonction

def ProiePredateur_test(Y, t, alpha_1, alpha_2, beta_1, beta_2):
    Yprime = np.zeros(2)
    Yprime[0] = alpha_1*Y[0] - beta_1*Y[0]*Y[1]
    Yprime[1] = -alpha_2*Y[1] + beta_2*Y[0]*Y[1]
    return Yprime


### Dans un premier temps, je fais varier les conditions initiales

Y0_T1 = np.array([10, 1])
test1 = odeint(ProiePredateur_test, Y0_T1, T_proie, (3, 2, 1, 1))

Y0_T2 = np.array([1,10])
test2 = odeint(ProiePredateur_test, Y0_T2, T_proie, (3, 2, 1, 1))

Y0_T3 = np.array([10,10])
test3 = odeint(ProiePredateur_test, Y0_T3, T_proie, (3, 2, 1, 1))


plt.plot(ProiePredateur_odeint[:,0], ProiePredateur_odeint[:,1], label = 'condtion exo')
plt.plot(test1[:,0], test1[:,1], label = 'proie >> predateur')
plt.plot(test2[:,0], test2[:,1], label = 'proie << predateur')
plt.plot(test3[:,0], test3[:,1], label = 'proie = predateur')
plt.title('Comportement des solutions en variant les conditions initiales')
plt.legend()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.grid()
plt.show()

## Les solutions restent stables, avec des amplitudes plus grandes, ce qui est normal puisqu'on part avec des populations plus denses

### Faisons varier cette fois les taux de reproduction alpha

test4 = odeint(ProiePredateur_test, Y0, T_proie, (10, 2, 1, 1))
test5 = odeint(ProiePredateur_test, Y0, T_proie, (3, 10, 1, 1))


plt.plot(ProiePredateur_odeint[:,0], ProiePredateur_odeint[:,1], label = 'condtion exo')
plt.plot(test4[:,0], test4[:,1], label = 'alph$a_1$ = 10')
plt.plot(test5[:,0], test5[:,1], label = 'alph$a_2$ = 10')
plt.title('Comportement des solutions en variant les taux de reproduction')
plt.legend()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.grid()
plt.show()

## En variant les taux de reproduction, les courbes sont soit très aplaties, soit très étendues vers le haut, ce qui est représentatif du problème puisque plus le taux est grand plus vite la population se multiplie 

### Faisons varier les coefficients de correlation beta

test6 = odeint(ProiePredateur_test, Y0, T_proie, (3, 2, 10, 1))
test7 = odeint(ProiePredateur_test, Y0, T_proie, (3, 2, 1, 10))

plt.plot(ProiePredateur_odeint[:,0], ProiePredateur_odeint[:,1], label = 'condtion exo')
plt.plot(test6[:,0], test6[:,1], label = 'bet$a_1$ = 10')
plt.plot(test7[:,0], test7[:,1], label = 'bet$a_2$ = 10')
plt.title('Comportement des solutions en variant les coefficients de couplage')
plt.legend()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.grid()
plt.show()

## Ce cas-ci est assez intéressant puisqu'on remarque que plus la relation entre les deux populations est forte, plus celle-ci va impacter leur développement. Ce qui donne lieu à une évolution instable de la solution


### Cette fois, on fait varier 2 paramètres à la fois c'est-à-dire, les conditions initiales et le taux de reproduction

Y0_T8 = np.array([10,1])
Y0_T9 = np.array([1,10])

test8 = odeint(ProiePredateur_test, Y0_T8, T_proie, (10, 2, 1, 1))
test9 = odeint(ProiePredateur_test, Y0_T9, T_proie, (3, 10, 1, 1))


plt.plot(ProiePredateur_odeint[:,0], ProiePredateur_odeint[:,1], label = 'condtion exo')
plt.plot(test8[:,0], test8[:,1], label = 'Y0 = [10, 1], alph$a_1$ = 10')
plt.plot(test9[:,0], test9[:,1], label = 'Y0 = [1, 10], alph$a_2$ = 10')
plt.title('Portraits de phase en modifiant deux paramètres')
plt.legend()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.grid()
plt.show()







