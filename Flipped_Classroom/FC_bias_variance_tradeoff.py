import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#from tqdm import tqdm


def comp_mc_return (state, num_step, gamma):
    g_t = 0

    for a in range(num_step):

        if random.uniform (0,1) > 0.5:
            if state ==1 :
                state = 0
            else: 

                state = 1
        g_t += state * (gamma**a)

    return g_t

def draw_estimation(episodes = 500):
    returns = []
    current_estimate =[]
     
    for i in range(episodes):

        returns.append(comp_mc_return(1,36,0.95))
        current_estimate.append(sum(returns)/len(returns))
    return current_estimate

iterations = 500

def task03():
    returns = []
    current_estimate =[]


    for i in range(iterations):
        
        for i in range(iterations):

            returns.append(draw_estimation(iterations))

            for i in len(returns):
                current_estimate[i] = (sum(returns[i])/len(returns))


def compute_td():
    






# plt.plot(range(1,iterations+1), current_estimate)
# plt.show()

df = pd.DataFrame(current_estimate, index =range(500))

mean = df.mean(axis =1)
std = df.std(axis =1)

ax = sns.lineplot(data = df, err_style='bars')
#ax.errorbar(df.index, mean, yerr = std)

plt.show()