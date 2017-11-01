import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.sparse import coo_matrix
import heapq
import itertools

# Clustering Problem
def generate_matrix(weight, obs_num):
    obs = []
    mean = [[0, 0], [3, 0], [0, 3]]
    cov = [[1, 0], [0, 1]]
    for i in range(len(mean)):
        obs.extend(np.random.multivariate_normal(mean[i], cov, np.int(weight[i] * obs_num)))
    return obs


# Calculate distance
def min_dis(x, u):
    distance = []
    for i in range(len(u)):
        distance.append(np.dot((x - u[i]), np.transpose(x - u[i])))
    return np.argmin(distance), np.min(distance)


#  Calculate the value of K-means objective function
def L_value(obs, k_means, u, L):
    # Initial the problem
    indicator = np.zeros((T, k_means))
    L_1 = L_2 = 0

    # given u, update c
    for i in range(len(obs)):
        temp_index, temp_distance = min_dis(obs[i], u)
        L_1 += temp_distance
        indicator[i, temp_index] = 1
    L.append(L_1)

    # given c, update u
    for i in range(len(u)):
        u[i] = np.dot(np.transpose(indicator[:, i]), obs) / np.sum(indicator[:, i])

    for i in range(len(obs)):
        u_star = np.dot(indicator[i, :], np.array(u))
        temp_distance = np.dot((obs[i] - u_star), np.transpose(obs[i] - u_star))
        L_2 += temp_distance
    L.append(L_2)

    return indicator


def K_means(obs):
    # Run and plot the result
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    for k_num in range(5, 1, -1):
        # initial the proble
        L = []
        u = []
        u_initial_index = np.random.choice(len(obs), k_num, replace=False)
        for i in range(k_num):
            u.append(obs[u_initial_index[i]])
        indicator = np.zeros((T, k_num))
        iterations = np.linspace(0.5, 20, 40)
        # start iteration
        for j in range(20):
            L_value(obs, k_num, u, L)
        plt.plot(iterations, L, label="k = " + str(k_num))
    axes.set_ylabel("L")
    axes.set_xlabel("Iterations")
    plt.legend(loc=1)
    plt.show()


def clustering_plot(k, obs):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    L = []
    u = []
    u_initial_index = np.random.choice(len(obs), k, replace=False)
    for i in range(k):
        u.append(obs[u_initial_index[i]])
    for j in range(20):
        indicator = L_value(obs, k, u, L)
    for i in range(len(indicator)):
        for j in range(1, k):
            indicator[i, 0] += indicator[i, j] * (j + 1)

    column_name = ["K"]
    for i in range(1, k):
        column_name.append("Non" + str(i))
    indicator = pd.DataFrame(indicator, columns=column_name)
    obs_new = pd.concat([pd.DataFrame(obs, columns=["x", "y"]), indicator], axis=1)
    groups = obs_new.groupby('K')
    for name, group in groups:
        plt.plot(group.x, group.y, marker="o", linestyle='', label=str(name))
    plt.legend()
    plt.show()


# Movie Recommendation
def preprocessing(d,lamda,theta_2):
   ratings = pd.read_csv('COMS4721_hw4-data/ratings.csv', header=None)
   ratings_test = pd.read_csv('COMS4721_hw4-data/ratings_test.csv', header=None)

   M = coo_matrix((ratings[2],(ratings[0],ratings[1]))).toarray()
   M_test = coo_matrix((ratings_test[2],(ratings_test[0],ratings_test[1]))).toarray()
   M = np.array(np.delete(np.delete(M, 0, 0),0,1))
   M_test = np.delete(np.delete(M_test, 0, 0),0,1)
   M_test = np.insert(M_test,len(M_test),0,axis=1)

   mean = np.zeros(d)
   cov = np.identity(d)/lamda
   V = np.array(np.transpose(np.random.multivariate_normal(mean, cov, max(ratings[1]))))
   U = np.array(np.ones((max(ratings[0]),d)))
   const = np.identity(d) * lamda * theta_2
   return M,M_test,V,U,const

def iteration(M,V,U,const,theta_2, lamda):
    # Updata U, V and record L
    L = []
    for i in range(100):
        for j in range(len(U)):
            index = np.ravel(np.nonzero(M[j,:]))
            temp_V = inv(const + np.dot(V[:,index],np.transpose(V[:,index])))
            U[j,:] = np.transpose(np.dot(temp_V,np.dot(V[:,index],np.transpose(M[j,index]))))

        for k in range(len(V[0])):
            index = np.ravel(np.nonzero(M[:,k]))
            temp_U = inv(const + np.dot(np.transpose(U[index,:]),U[index,:]))
            V[:,k] = np.dot(temp_U,np.dot(np.transpose(U[index,:]),M[index,k]))

        index = np.nonzero(M)
        part1 = np.sum((M[index] - np.dot(U,V)[index]) ** 2)/(2 * theta_2)
        part2 = np.sum(U ** 2) * lamda / 2.0
        part3 = np.sum(V ** 2) * lamda / 2.0
        L.append(-(part1+part2+part3))
    return L, np.dot(U,V), U, V

def RMSE(runtimes):
    d = 10
    lamda = 1
    theta_2 = 0.25
    n = np.linspace(1,100,100)
    table = []
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    for i in range(runtimes):
        M, M_test,V,U,const = preprocessing(d, lamda, theta_2)
        L, M_predict, U, V = iteration(M, V,U,const, theta_2, lamda)
        indice = np.nonzero(M_test)
        RMSE = np.sqrt(np.sum(np.array(M_test[indice] - M_predict[indice])**2) / np.count_nonzero(M_test))
        table.append([RMSE, L[-1], U, V])
        plt.plot(n[1:], L[1:], label="times = " + str(i))
    table_f = pd.DataFrame(table, columns=('RMSE','L[-1]','U','V')).sort(['L[-1]'], ascending = False)
    table_f.to_pickle('Result.pkl')
    plt.legend()
    plt.show()

def map(nearest, dis):
    for j in range(len(dis[0])):
        while nearest == dis[0, j]:
            return dis[1,j]

def recommendation():
    # preprocessing dataset
    movie_name = ['Star Wars','My Fair Lady','GoodFellas']
    movies = pd.read_csv('COMS4721_hw4-data/movies.txt', sep="\n",header= None)
    movies[1] = movies.index
    result = pd.read_pickle('Result.pkl')
    V = result.loc[8,'V']

    for i in range(len(movie_name)):
        index = movies[movies[0].str.contains(movie_name[i])][1]
        dis = np.zeros((2,len(V[0])))

        for j in range(len(V[0])):
            dis[0,j] = np.sum(np.array(np.ravel(V[:,index]) - V[:,j]) ** 2)
            dis[1,j] = j

        nearest = heapq._nsmallest(11,dis[0,:])
        print(nearest)
        choice = []
        for k in range(len(nearest)):
            choice.append(map(nearest[k],dis))
        print(movies.loc[choice,0])


if __name__ == "__main__":
    # Generate data with weight
    weight = [0.2, 0.5, 0.3]
    T = 500
    obs = generate_matrix(weight, T)
    K_means(obs)
    clustering_plot(3, obs)
    P1_Q2(5, obs)
    runtimes = 10
    RMSE(runtimes)
    recommendation()





