import numpy as np
import matplotlib.pyplot as plt
import colour as cl

def color_hue(vector_color):
    x = vector_color[0]
    y = vector_color[1]
    return np.arctan((y/x))

def color_hue_2(vector_color):
    x = vector_color[0]
    y = vector_color[1]
    return 2*np.arctan((y/x))

def coloreando_V(XY, n, a, b, beta, modelo):
    l = 2*n+1
    plt.figure(figsize=(a,b))
    for i in range(l):
        for j in range(l):
            color_vector = XY[i, j,:]
            color = cl.Color(hue=color_hue_2(color_vector), saturation=1, luminance=0.3)
            plt.plot(i,j, marker="o", color=str(color))
    plt.title("Visualización de " + modelo +" a beta igual " +str(beta))
    plt.show()

def coloreando(XY, n, a, b, beta, modelo):
    l = 2*n+1
    plt.figure(figsize=(a,b))
    for i in range(l):
        for j in range(l):
            colorH = XY[i, j]
            color = cl.Color(hue=colorH, saturation=1, luminance=0.3)
            plt.plot(i,j, marker="o", color=str(color))
    plt.title("Visualización de " + modelo +" a beta igual " +str(beta))
    plt.show()

def viendo_el_grafo(M_v, M_h, n,a,b):
    #M_v es de lxl+1 y M_h es de l+1x1
    l =2*n+1
    plt.figure(figsize=(a,b))
    for fila in range(l):
        for columna in range(l):
            #horizontales
            if columna<(l-1):
                if M_h[fila,columna] == 1:
                    y = (fila, fila)
                    x = (columna, columna+1)
                    plt.plot(x,y, marker = "o", color = "black")
            #verticales
            if fila<(l-1):
                if M_v[fila,columna] == 1:
                    y = (fila, fila+1)
                    x = (columna, columna)
                    plt.plot(x,y, marker = "o", color = "black")
            #else:
              #  plt.plot(fila,columna, marker ="o", color= "black")
            plt.plot(fila,columna, marker ="o", color= "black")
    plt.title("Rejilla a aristas abiertas y cerradas dadas")
    plt.show()

def clusters_grafo(M_v, M_h, n,a,b, XY):
    l =2*n+1
    fig, ax = plt.subplots(1,2, figsize = (a,b))
    for fila in range(l):
        for columna in range(l):
            #horizontales
            if columna<(l-1):
                if M_h[fila,columna] == 1:
                    y = (fila, fila)
                    x = (columna, columna+1)
                    ax[0].plot(x,y, marker = "o", color = "black")
            #verticales
            if fila<(l-1):
                if M_v[fila,columna] == 1:
                    y = (fila, fila+1)
                    x = (columna, columna)
                    ax[0].plot(x,y, marker = "o", color = "black")
            ax[0].plot(fila,columna, marker ="o", color= "black")

    for i in range(l):
        for j in range(l):
            #color_H = XY[i, j]
            color = cl.Color(hue= XY[i,j], saturation=1, luminance=0.5)
            ax[1].plot(i,j, marker="o", color=str(color))
    plt.title("grafo despues de clusterización y observación v.a.")
    plt.show()

def coloreando2(rotor, n, a, b):
    l = 2*n
    plt.figure(figsize=(a,b))
    for i in range(l):
        for j in range(l):
            num = rotor[i,j]
            valor = int(num)
            if valor>0:
                plt.plot(i, j, marker ="o", color = "grey" )
            elif valor<0:
                plt.plot(i, j, marker ="o", color = "black" )

    plt.title("Visualización del rotor para beta igual a ")
    plt.show()

def Visualizando(Villain, rotor, n, a, b, beta):
    l = 2*n+1 #3 0 1 2
    rho = 2*l-1 # 5 0.. 4
    plt.figure(figsize=(a,b))
    for i in range(rho):
        for j in range(rho):
            if i%2 == 0 and j%2 == 0:
                l = int(i/2)
                k = int(j/2)
                color_vector = Villain[l, k]
                color = cl.Color(hue=color_vector, saturation=1, luminance=0.5)
                plt.plot(i,j, marker="o", color=str(color))
            elif i%2 !=0  and j%2 != 0:
                m = int((i-1)/2)-2
                n = int((j-1)/2)-2
                num = rotor[m,n]
                valor = int(num)
                if valor>0:
                    plt.plot(i,j, marker ="o", color = "black" )
                elif valor<0:
                    plt.plot(i,j, marker ="o", color = "gray" )
    plt.title("Valor obtenido para beta igual "+str(beta))
    plt.show()



def todos_ordenados(n):
    l = 2*n+1
    x = np.zeros((l, l, 2))
    for i in range(l):
        for j in range(l):
            x[i,j, 0] = 1
    return x

def rotacion(u):
    a = np.cos(u)
    b = np.sin(u)

    R = np.array([[a, -b],[b, a]])
    return R

def promV(x, i ,j, n):
    l = 2*n
    v_x = [x[i,j+1*(j<l), :]*(j<l), x[i,j-1*(j>0), :]*(j>0), x[i+1*(i<l),j, :]*(i<l), x[i-1*(i>0),j, :]*(i>0)]
    d_x = 1*(j<l) + 1*(0<j) + 1*(0<i) + 1*(i<l)
    return (1/d_x)*sum(v_x)

def resta(x,y, i, j, n):
    l=2*n
    v_x = [x[i,j+1*(j<l), :], x[i,j-1*(j>0), :], x[i+1*(i<l),j, :], x[i-1*(i>0),j, :]]
    vx = 0
    v_y = [y[i,j+1*(j<l), :], y[i,j-1*(j>0), :], y[i+1*(i<l),j, :], y[i-1*(i>0),j, :]]
    vy=0
    for vecinoY in v_y:
        vy+=np.linalg.norm(y[i,j,:] - vecinoY)**2
    for vecinoX in v_x:
        vx+=np.linalg.norm(x[i, j,:] - vecinoX)**2
    return vy-vx

def gibbs_XY(x, y, i, j,n, beta):
    return np.exp( (-beta/2)*resta(x, y, i,j ,n) )

def q(x, y, i ,j, n, beta):
    l = 2*n
    v_x = [x[i,j+1*(j<l), :]*(j<l), x[i,j-1*(j>0), :]*(j>0), x[i+1*(i<l),j, :]*(i<l), x[i-1*(i>0),j, :]*(i>0)]
    d_x = 1*(j<l) + 1*(0<j) + 1*(i<l) + 1*(0<i)
    c_x = (1/d_x)*sum(v_x)
    r_x =np.linalg.norm(c_x)
    vec = y[i, j,:]
    x = vec[0]
    y = vec[1]
    theta = 2*np.arctan((y/x))
    return np.exp((-1)*d_x*beta*((theta/r_x)**2))

def q_2(x, y, i, j, n, beta):
    l = 2*n
    v_x = [x[i,j+1*(j<l), :]*(j<l), x[i,j-1*(j>0), :]*(j>0), x[i+1*(i<l),j, :]*(i<l), x[i-1*(i>0),j, :]*(i>0)]
    d_x = 1*(j<l) + 1*(0<j) + 1*(i<l) + 1*(0<i)
    c = (1/d_x)*sum(v_x)
    return np.exp(d_x*beta*0.5*(np.linalg.norm(y[i, j,:] - c)**2))

def postula(x, n, beta):
    y = np.copy(x)
    l =2*n+1
    i = np.random.randint(0, l)
    j = np.random.randint(0, l)
    c = promV(x, i, j,n)
    r = np.linalg.norm(c)
    c_1 = (r**(-1))*c
    theta = 100
    sigma =np.sqrt((1/(r*beta)))
    while (theta>np.pi) or (theta< -1*np.pi):
        theta = np.random.normal(loc = 0, scale = sigma)
    y[i, j,:] = rotacion(theta)@c_1
    return (i, j, y)

def Metropolis_Hasting_XY(N, n, x_0, beta):
    x = x_0
    U = np.random.uniform(size=N)
    for k in range(N):
        u = U[k]
        i, j , y = postula(x, n, beta)
        c = gibbs_XY(x, y, i, j,n, beta)
        Q = q(x,y, i, j, n,beta)/q(y, x, i, j, n,beta)
        alpha = c*Q
        if alpha >= u:
            x = y
        else:
            x = x
    return x



def g_aux(k, p):
    if k>=1:
        return (p/2)*((1-p)**(k-1))
    elif k<=0:
        r = np.abs(k)
        return (p/2)*((1-p)**r)

def x_a(p):
    sigma = np.random.binomial(1, 0.5)
    G = np.random.geometric(p)
    if sigma == 0:
        return G
    else:
        return -1*(G-1)


def g2_aux(k, p):
    if k>0:
        return (p/2)*((1-p)**(k-1))
    elif k<=0:
        r = np.abs(k)

def g_sinC(k, p, beta, mu):
    efe = np.exp(((-beta)/2)*((k-mu)**2))
    value = g_aux(k,p)
    return efe/value

def g2_sinC(k, p, beta, mu):
    efe = np.exp(((-beta)/2)*((k-mu)**2))
    value = g2_aux(k,p)
    return efe/value

def pol(x, p, beta,mu):
     return (beta/2)*((x-mu)**2)+(np.abs(x)*np.log(1-p))

def polinomio(x,p, beta, mu):
    value = pol(x, p, beta,mu)
    return (2/p)*np.exp((-1)*value)


def g_mu(k, p, beta, mu):
    x_n = 0
    x_1 = (beta*mu-np.log(1-p))/beta
    x_2 = (beta*mu+np.log(1-p))/beta
    a1 = polinomio(x_2, p, beta, mu)
    a2 = polinomio(x_n, p, beta, mu)
    a3 = polinomio(x_1, p, beta, mu)
    c = max(a1, a2, a3)
    efe = np.exp(((-beta)/2)*((k-mu)**2))
    value = g_aux(k, p)
    return (efe/value)*(1/c)

def IvG_mu(beta,mu):
    p = 0.3
    k = x_a(p)
    u = np.random.uniform()
    p_ent = np.floor(mu)
    mu_t = mu-p_ent
    while u>g_mu(k, p, beta, mu_t):
        k = x_a(p)
        u = np.random.uniform()
    return k + p_ent

def Serie_LHS(x, t):
    S1 = np.array([-2,-1,0,1,2])
    return np.sqrt(t)*sum([np.exp(-1*t*np.pi*((x+n)**2)) for n in S1])

def Serie_RHS(x, t):
    S1 = np.array([1,2])
    return 1+2*sum([np.exp(-1*np.pi*(t**(-1))*(n**2))*np.cos(2*np.pi*n*x) for n in S1])

def f(x, beta):
    Tau = 2*np.pi*beta
    t = 1/beta
    if t<2*np.pi:
        return Serie_LHS(x/(2*np.pi), Tau)/np.sqrt(Tau)
    elif t>=2*np.pi:
        return  Serie_RHS(x/(2*np.pi),Tau)/np.sqrt(Tau)

def g(theta_1, theta_2, beta):
    x = theta_1-theta_2
    y = theta_1+theta_2-np.pi
    return (f(x,beta)-f(y,beta))/f(x, beta)

def cambio_eje(x,nu):
    y=np.copy(x)
    mod = 2*np.pi
    return (y + nu)%mod

def cambio_eje_b0(x,nu,n):
    l = 2*n+1
    y=np.copy(x)
    mod = 2*np.pi
    for i in range(1,l-1):
        for j in range(1,l-1):
            theta = y[i,j]
            theta_R = (theta+nu)%mod
            y[i,j] = theta_R
    return y

def Aristas_V(n,x_0,beta, nu):
    l=2*n
    M_v = np.zeros((l,l+1))
    mod = 2*np.pi
    for fila in range(l):
        for columna in range(l+1):
            u = np.random.uniform(low=0, high=1)
            theta_1= x_0[fila, columna]
            theta_2= x_0[fila+1, columna]
            angle_1 = (theta_1-nu)%mod
            angle_2 = (theta_2-nu)%mod
            if (np.cos(angle_1)>0) and (np.cos(angle_2)>0):
                if u<=g(angle_1,angle_2, beta):
                    M_v[fila, columna] = 1
            if (np.cos(angle_1)<0) and (np.cos(angle_2)<0):
                angle_1N = (np.pi + angle_1)%mod
                angle_2N = (np.pi + angle_2)%mod
                if u<=g(angle_1N,angle_2N, beta):
                    M_v[fila, columna] = 1
    return M_v


def Aristas_H(n,x_0,beta,nu):
    l=2*n
    M_h = np.zeros((l+1,l))
    mod = 2*np.pi
    for fila in range(l+1):
        for columna in range(l):
            u2 = np.random.uniform(low=0, high=1)
            theta_1= x_0[fila, columna]
            theta_2= x_0[fila, columna+1]
            angle_1 = (theta_1-nu)%mod
            angle_2 = (theta_2-nu)%mod
            if (np.cos(angle_1)>0) and (np.cos(angle_2)>0):
                if u2<=g(angle_1,angle_2, beta):
                    M_h[fila, columna] = 1
            if (np.cos(angle_1)<0) and (np.cos(angle_2)<0):
                angle_1N = (np.pi+angle_1)%mod
                angle_2N = (np.pi+angle_2)%mod
                if u2<=g(angle_1N, angle_2N, beta):
                    M_h[fila, columna] = 1 # np.random.binomial(1, g(angle_1N, angle_2N, beta)) #open
    return M_h

def vecinos(fila, columna, n, M_v, M_h):
    l =2*n+1
    vecinos = []
    #columns
    if 0<columna<(l-1):
        if M_h[fila, columna] == 1:
            vecinos.append((fila, columna+1))
        if M_h[fila, columna-1] == 1:
            vecinos.append((fila, columna-1))
    elif columna == l-1:
        if M_h[fila, columna-1] == 1:
            vecinos.append((fila, columna-1))
    elif columna == 0:
        if M_h[fila, columna]== 1:
            vecinos.append((fila,columna+1))

    #rows
    if 0<fila<(l-1):
        if M_v[fila, columna] == 1:
            vecinos.append((fila+1, columna))
        if M_v[fila-1, columna] == 1:
            vecinos.append((fila-1, columna))
    elif fila == l-1:
        if M_v[fila-1, columna] == 1:
            vecinos.append((fila-1, columna))
    elif fila == 0:
        if M_v[fila, columna] ==1:
            vecinos.append((fila+1, columna))
    return vecinos

def connected_components(n, M_v,M_h):
    l = 2*n+1
    Visited = np.zeros((l,l))
    Cc = []
    for fila in range(l):
        for columna in range(l):
            if Visited[fila,columna] == 0:
                cc = [(fila,columna)]
                Visited[fila,columna] = 1
                q = [(fila,columna)]
                while q!=[]:
                    w = q[0]
                    q.remove(w)
                    chequeo = vecinos(w[0], w[1], n, M_v, M_h)
                    for k in chequeo:
                        if Visited[k] == 0:
                            q.append(k)
                            cc.append(k)
                            Visited[k] = 1
                Cc.append(cc)
    return Cc

def Rotar(M_v,M_h, n, x, nu):
    mod = 2*np.pi
    Cc = connected_components(n, M_v,M_h)
    for cc in Cc:
        a = np.random.binomial(1, 0.5)
        if a == 1:
            for v in cc:
                x[v] = x[v]
        else:
            for v in cc:
                x[v] = (2*nu + np.pi - x[v])%mod
    return x

def Swendsen_Wang_Villain(N, radius,beta, x0 =str(1)):
    l = 2*radius+1
    if x0 == str(1):
        x0 = np.zeros((l,l))
    else:
        x0 = x0
    x = np.copy(x0)
    while N!=0:
        nu = np.random.uniform(low=0, high=2*np.pi)
        Mh_S = Aristas_H(radius,x,beta,nu)
        Mv_S = Aristas_V(radius,x,beta,nu)
        comp_connect = connected_components(radius,Mv_S,Mh_S)
        if len(comp_connect) != 1:
            x = Rotar(Mv_S, Mh_S, radius, x, nu)
        N = N-1
    return x

def Swendsen_Wang_Villain_H(N, radius,beta, x0 =str(1)):
    l = 2*radius+1
    if x0 == str(1):
        x0 = np.zeros((l,l))
    else:
        x0 = x0
    x = np.copy(x0)
    CC = []
    while N!=0:
        nu = np.random.uniform(low=0, high=2*np.pi)
        Mh_S = Aristas_H(radius,x,beta,nu)
        Mv_S = Aristas_V(radius,x,beta,nu)
        y = np.copy(x)
        comp_connect = connected_components(radius,Mv_S,Mh_S)
        if len(comp_connect) != 1:
            x = Rotar(Mv_S, Mh_S, radius, x,nu)
        CC.append((Mh_S, Mv_S, y, x, nu))
        N = N-1
    return (x, CC)

def is_in_boundary(cc,n):
    l = 2*n
    for v in cc:
        if v[0] == 0 or v[0] == l or v[1] == 0 or v[1] == l:
            return 1 # 1 will mean i am in the boundary
    else:
        return 0 # 0 means the vertex don't touch the boundary

def Rotar_BC(M_v, M_h, n, x_0, nu):
    x = np.copy(x_0)
    mod = 2*np.pi
    CC = connected_components(n, M_v, M_h)
    for cc in CC:
        if is_in_boundary(cc,n) == 0:
            a = np.random.binomial(1, 0.5) # a number in {0,1}
            if a == 1:
                for v in cc:
                    x[v] = x[v]
            else:
                for v in cc:
                    x[v] = (2*nu + np.pi - x[v])%mod
    return x

def Swendsen_Wang_Villain_b0(N,n,beta, x_0 = str(1)):
    l = 2*n+1
    if x_0 == str(1):
        x_0 = np.zeros((l,l))
    x = np.copy(x_0)
    while N!=0:
        nu = np.random.uniform(low=0, high=2*np.pi)
        M_h = Aristas_H(n,x,beta,nu)
        M_v = Aristas_V(n,x,beta,nu)
        x = Rotar_BC(M_v, M_h, n, x, nu)
        N = N-1
    return x

def Ka_v(theta, n):
    l = 2*n+1
    mu_v = np.zeros((l-1, l))
    mu_h = np.zeros((l, l-1))

    for i in range(l-1):
        mu_v[i, :] = theta[:, i] - theta[:, i+1]
        mu_h[:, i] = -theta[i, :] + theta[i+1, :]
    c = 1/(2*np.pi)
    return c*mu_v, c*mu_h

def var_k(n,theta,beta):
    l =2*n+1
    k_v = np.zeros((l-1,l))
    k_h = np.zeros((l,l-1))
    mu_s = Ka_v(theta,n)
    mu_v_0 = mu_s[0]
    mu_h_0 = mu_s[1]
    for i in range(l-1):
        for j in range(l):
            k_v[i,j] = IvG_mu(beta*((2*np.pi)**2), mu_v_0[i,j])
            k_h[j,i] = IvG_mu(beta*((2*np.pi)**2), mu_h_0[j,i])

    return k_v, k_h

def rotores(K_v, K_h,n):
    l = 2*n+1
    rotor = np.zeros((l-1,l-1))
    for i in range(l-1):
        for j in range(l-1):
            rotor[i,j] = ((-1)*K_v[i,j] + K_v[i,j+1] )+ (K_h[i,j] + (-1)*K_h[i+1,j])
    return rotor

def gas_de_coulumb(N, n, beta):
    theta = Swendsen_Wang_Villain_H(N, n, beta)
    auxiliar = var_k(n,theta,beta)
    k_v = auxiliar[0]
    k_h = auxiliar[1]
    rotor = rotores(k_v, k_h, n)
    return theta, rotor

def gas_de_coulumb_H(N, n, beta):
    theta = Swendsen_Wang_Villain_H(N, n, beta)
    auxiliar = var_k(n,theta,beta)
    k_v = auxiliar[0]
    k_h = auxiliar[1]
    rotor = rotores(k_v, k_h, n)
    return theta, rotor, auxiliar
