# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:54:51 2022

@author: fabio
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import os

directory = os.path.dirname(os.path.realpath(__file__)) + '\\'
weights_and_biases_folder = r'weights_and_biases\\'
fig_folder = r'Figures'
run = 3

if os.path.exists(directory + weights_and_biases_folder + '{:03d}'.format(run)):
    pass
else:
    os.mkdir(directory + weights_and_biases_folder + '{:03d}'.format(run))

terrain_size = 20

mutation_rate = 0.9
generations = 40
ants_per_generation = 100
elitism = max(int(0.15*ants_per_generation), 1)

steps_per_run = 100
avoid_walls_perimeter = True

draw_iterations = False
draw_every_n_iterations = 1

parameters_file = 'Parameters.txt'

f = open(directory + weights_and_biases_folder + '{:03d}'.format(run) + '\\' + 
         parameters_file, 'w')
f.write('terrain_size: ' + str(terrain_size) + '\n')
f.write('mutation_rate: ' + str(mutation_rate) + '\n')
f.write('generations: ' + str(generations) + '\n')
f.write('ants_per_generation: ' + str(ants_per_generation) + '\n')
f.write('elitism: ' + str(elitism) + '\n')
f.write('steps_per_run: ' + str(steps_per_run) + '\n')
f.write('avoid_walls_perimeter: ' + str(avoid_walls_perimeter) + '\n')

f.close()

def generate_food(terrain_size):
    food = np.random.randint(1, terrain_size - 1, 2)
    return food


# get name of variable as string f'{ant=}'.split('=')[0]
# np.save(file, array) per salvare np.array
# esempio: np.save(directory + weights_and_biases_folder + '\\' + 
# f'{wih=}'.split('=')[0], wih)
# np.load(file) per ricaricare il file

def save_weighs_and_biases(w_i_h, w_h_o, b_h, b_o, generation):
    if os.path.exists(directory + weights_and_biases_folder + \
                      '{:03d}'.format(run) + '\\' + \
                          '{:04d}'.format(generation)):
        pass
    else:
        os.mkdir(directory + weights_and_biases_folder + \
                  '{:03d}'.format(run) + '\\' +\
                      '{:04d}'.format(generation))
    
    
    file_path = directory + weights_and_biases_folder + \
                      '{:03d}'.format(run) + '\\' + \
                          '{:04d}'.format(generation)
    
    file_list = [w_i_h, w_h_o, b_h, b_o]
    file_names = ['w_i_h', 'w_h_o', 'b_h', 'b_o']
    
    for i in range(len(file_list)):
        file_name = file_names[i]
        np.save(file_path + '\\' + file_name, file_list[i])

def load_weights_and_biases(generation):
    
    file_path = directory + weights_and_biases_folder + \
                      '{:03d}'.format(run) + '\\' + \
                          '{:04d}'.format(generation)
    
    file_list = []
    file_names = ['w_i_h', 'w_h_o', 'b_h', 'b_o']
    
    for i in range(len(file_names)):
        file_name = file_names[i]
        file_list.append(np.load(file_path + '\\' + file_name + '.npy'))
    
    return file_list
        
def load_rewards(selected_run = run):
    file_path = directory + weights_and_biases_folder + \
                      '{:03d}'.format(selected_run) + '\\'
    
    rewards = np.load(file_path + 'rewards.npy')
    return rewards
        
def g (x):
    return 1/(1 + math.e**(-x))

"""
Per usare atan2 (y,x) fare rotazione vettore food in modo da allineare a 
ant_direction
"""
def G_matrix(ant_direction):
    v1 = ant_direction
    v2 = np.array([1,0])
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    
    cos_angle = np.dot(unit_vector_1, unit_vector_2)
    sin_angle = np.cross(unit_vector_1, unit_vector_2)
    
    return np.array([[cos_angle, -sin_angle],[sin_angle, cos_angle]])

#G = G_matrix(ant_direction)    
#vec = G.dot(v2) con v2 = v2 = food_position - ant_position 
# np.arctan2(vec[1], vec[0])
# np.arctan2(vec[1], vec[0])*180/math.pi

def angle_ant_food(ant_direction, ant_position, food_position):
    if (food_position == ant_position).all():
        angle = 0
    else:
        v2 = food_position - ant_position 
        unit_vector_2 = v2 / np.linalg.norm(v2)
        G = G_matrix(ant_direction) 
        vec = G.dot(unit_vector_2)
        angle = np.arctan2(vec[1], vec[0])
    return angle


def running_ant(ant,food,w_in_h, w_h_o, b_h, b_o, steps, terrain_size):
    reward = 0
    for j in range(steps):
        
        distance = np.linalg.norm(ant[:2]-food) / (terrain_size * math.sqrt(2))
        angle = angle_ant_food(ant[2:], ant[:2], food)
        
        #input_layer = np.append(np.append(ant[:2],food)/100,distance)
        input_layer = np.append(distance, angle)
        hidden_layer = np.zeros(4,)
        output_layer = np.zeros(3,)
        
        for ii in range(len(hidden_layer)):
            hidden_layer[ii] = g(np.dot(input_layer, w_in_h[:,ii]) + b_h[ii])
            for i in range(len(output_layer)):
                output_layer[i] = g(np.dot(hidden_layer, w_h_o[:,i]) + b_o[i])
        
        index_max = np.argmax(output_layer)
        new_ant_direction = np.zeros(2,)
        if index_max == 0:
            new_ant_direction[0] = ant[2]
            new_ant_direction[1] = ant[3]
        if index_max == 1:
            new_ant_direction[0] = ant[3]
            new_ant_direction[1] = -ant[2]
        if index_max == 2:
            new_ant_direction[0] = -ant[3]
            new_ant_direction[1] = ant[2]
        
        new_ant_position = ant[:2] + new_ant_direction
        new_ant = np.append(new_ant_position, new_ant_direction) 
        
        if ((new_ant_position >= terrain_size).any() or \
            (new_ant_position <= 0).any()) and avoid_walls_perimeter:
            new_ant_position = np.random.randint(1,terrain_size-1,2)
            new_ant_direction  = np.array(direction_list[np.random.randint(4)])
            new_ant = np.append(new_ant_position, new_ant_direction) 
            reward = reward - 1
            
        flag = 0
        while np.array_equal(food,new_ant_position):
            food = generate_food(terrain_size)
            flag = 1
            
        if flag == 1:
            reward = reward + 1
        
        ant = new_ant
        
        # plt.scatter([new_ant_position[0],food[0]],[new_ant_position[1],food[1]],color=['r','b'])
        # # fig.canvas.update()
        # plt.xlim([0,100])
        # plt.ylim([0,100])
        # plt.grid(visible=True)
        # # fig.canvas.flush_events()
        # plt.show()
        # #plt.pause(0.1) 
        
    return ant, reward, food

def see_run(steps, terrain_size, generation, selected_run = run):    
    food = generate_food(terrain_size)
    rewards = load_rewards(selected_run)
    index_reward_max = np.argmax(rewards[:,generation])

    ant_position = np.random.randint(1,terrain_size-1,2)
    ant_direction  = np.array(direction_list[np.random.randint(4)])
    ant = np.append(ant_position, ant_direction)

    w_input_hidden_list, w_hidden_output_list, hidden_bias_list, \
        output_bias_list =  load_weights_and_biases(generation)

    w_in_h = w_input_hidden_list[:,:,index_reward_max]
    w_h_o = w_hidden_output_list[:,:,index_reward_max]
    b_h = hidden_bias_list[:,index_reward_max]
    b_o = output_bias_list[:,index_reward_max]
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter([ant[0],food[0]], [ant[1],food[1]], color=['r','b'])
    plt.xlim([0,terrain_size])
    plt.ylim([0,terrain_size]) 
    
    for j in range(steps):
        
        distance = np.linalg.norm(ant[:2]-food) / (terrain_size * math.sqrt(2))
        angle = angle_ant_food(ant[2:], ant[:2], food)
        
        #input_layer = np.append(np.append(ant[:2],food)/100,distance)
        input_layer = np.append(distance, angle)
        hidden_layer = np.zeros(4,)
        output_layer = np.zeros(3,)
        
        for ii in range(len(hidden_layer)):
            hidden_layer[ii] = g(np.dot(input_layer, w_in_h[:,ii]) + b_h[ii])
            for i in range(len(output_layer)):
                output_layer[i] = g(np.dot(hidden_layer, w_h_o[:,i]) + b_o[i])
        
        index_max = np.argmax(output_layer)
        new_ant_direction = np.zeros(2,)
        if index_max == 0:
            new_ant_direction[0] = ant[2]
            new_ant_direction[1] = ant[3]
        if index_max == 1:
            new_ant_direction[0] = ant[3]
            new_ant_direction[1] = -ant[2]
        if index_max == 2:
            new_ant_direction[0] = -ant[3]
            new_ant_direction[1] = ant[2]
        
        new_ant_position = ant[:2] + new_ant_direction
        new_ant = np.append(new_ant_position, new_ant_direction)
        
        plt.scatter([food[0],new_ant_position[0]],
                    [food[1], new_ant_position[1]],color=['b','r'])
        # fig.canvas.update()
        plt.xlim([0,terrain_size])
        plt.ylim([0,terrain_size])
        plt.grid(visible=True)
        plt.title('Generation ' + str(generation))
        # fig.canvas.flush_events()
        plt.show()
        # plt.pause(0.1) 
        
        if ((new_ant_position >= terrain_size).any() or \
            (new_ant_position <= 0).any()) and avoid_walls_perimeter:
            new_ant_position = np.random.randint(1,terrain_size-1,2)
            new_ant_direction  = np.array(direction_list[np.random.randint(4)])
            new_ant = np.append(new_ant_position, new_ant_direction) 
            
        while np.array_equal(food,new_ant_position):
            food = generate_food(terrain_size)
        
        ant = new_ant
    
def random_reproduction(w_i_h_list, w_h_o_list, b_h_list, b_o_list, 
                 mutation_rate, best_n, generation):
    
    indices_sort = np.flip(np.argsort(rewards[:,generation]))
    parent_index = indices_sort[np.random.randint(0, best_n + 1)]
    
    mutation_weights_input_hidden = np.random.uniform(low=-mutation_rate, 
                                                      high=mutation_rate, 
                                                      size=(2,4))
    
    mutation_weights_hidden_output = np.random.uniform(low=-mutation_rate, 
                                                       high=mutation_rate, 
                                                       size=(4,3))
    
    mutation_bias_hidden = np.random.uniform(low=-mutation_rate, 
                                             high=mutation_rate, 
                                             size=(4,))
    
    mutation_bias_output = np.random.uniform(low=-mutation_rate, 
                                             high=mutation_rate, 
                                             size=(3,))

    weight_input_hidden = w_i_h_list[:,:,parent_index] \
        + mutation_weights_input_hidden
    weight_hidden_output = w_h_o_list[:,:,parent_index] \
        + mutation_weights_hidden_output
    bias_hidden = b_h_list[:,parent_index] \
        + mutation_bias_hidden
    bias_output = b_o_list[:,parent_index] \
        + mutation_bias_output
    
    return weight_input_hidden, weight_hidden_output, bias_hidden, bias_output

def crossover_reproduction(w_i_h_list, w_h_o_list, b_h_list, b_o_list, 
                 mutation_rate, best_n, generation):
    
    # sorted_rewards = np.flip(np.sort(rewards[:,generation]))
    indices_sort = np.flip(np.argsort(rewards[:,generation]))
    
    parent1_index = indices_sort[np.random.randint(0, best_n + 1)]
    parent2_index = indices_sort[np.random.randint(0, best_n + 1)]
    while parent2_index == parent1_index:
        parent2_index = indices_sort[np.random.randint(0, best_n + 1)]
    
    w_i_h_child = np.zeros(w_i_h_list[:,:,0].shape)
    for i in range(w_i_h_child.shape[0]):
        for j in range(w_i_h_child.shape[1]):
            w_i_h_child[i,j] = np.random.choice(\
            [w_i_h_list[i,j,parent1_index],\
             w_i_h_list[i,j,parent2_index]])
    
    w_h_o_child = np.zeros(w_h_o_list[:,:,0].shape)
    for i in range(w_h_o_child.shape[0]):
        for j in range(w_h_o_child.shape[1]):
            w_h_o_child[i,j] = np.random.choice(\
            [w_h_o_list[i,j,parent1_index],\
             w_h_o_list[i,j,parent2_index]])
    
    b_h_child = np.zeros(b_h_list[:,0].shape)
    for i in range(b_h_child.shape[0]):
        b_h_child[i] = np.random.choice([b_h_list[i,parent1_index],\
                                        b_h_list[i,parent2_index]])
    
    b_o_child = np.zeros(b_o_list[:,0].shape)
    for i in range(b_o_child.shape[0]):
        b_o_child[i] = np.random.choice([b_o_list[i,parent1_index],\
                                        b_o_list[i,parent2_index]])
    
    return w_i_h_child, w_h_o_child, b_h_child, b_o_child
    

def copy_parent(w_i_h_list, w_h_o_list, b_h_list, b_o_list, nth, generation):
    
    indices_sort = np.flip(np.argsort(rewards[:,generation]))
    parent_index = indices_sort[nth]

    w_i_h_child = w_i_h_list[:,:,parent_index]
    w_h_o_child = w_h_o_list[:,:,parent_index]
    b_h_child = b_h_list[:,parent_index]
    b_o_child = b_o_list[:,parent_index]
    
    return w_i_h_child, w_h_o_child, b_h_child, b_o_child

#call a random function with argument
# list_of_functions = [f1, f2, f3]

# result = random.choice(list_of_functions)('abc')
    
def generate_population(w_i_h_list, w_h_o_list, b_h_list, b_o_list, elite, 
                        generation, mutation_rate):
    w_i_h_gen_list = np.zeros(w_i_h_list[:,:,:].shape)
    w_h_o_gen_list = np.zeros(w_h_o_list[:,:,:].shape)
    b_h_gen_list = np.zeros(b_h_list[:,:].shape)
    b_o_gen_list = np.zeros(b_o_list[:,:].shape)
    for k in range(w_i_h_gen_list.shape[2]):
        if k < elite:
            w_i_h_gen_list[:,:,k], w_h_o_gen_list[:,:,k], b_h_gen_list[:, k], \
                b_o_gen_list[:, k] = copy_parent(w_i_h_list, w_h_o_list, \
                                                 b_h_list, b_o_list, k, \
                                                     generation)
        else:
            list_of_functions = [crossover_reproduction(w_i_h_list, 
                                                        w_h_o_list, 
                                                        b_h_list, 
                                                        b_o_list,
                                                        mutation_rate, 
                                                        elite,
                                                        generation), 
                                 random_reproduction(w_i_h_list, 
                                                     w_h_o_list, 
                                                     b_h_list, 
                                                     b_o_list, 
                                                     mutation_rate,
                                                     elite, 
                                                     generation)]
            
            w_i_h_gen_list[:,:,k], w_h_o_gen_list[:,:,k], b_h_gen_list[:, k], \
                b_o_gen_list[:, k] = random.choice(list_of_functions)
            
    return w_i_h_gen_list, w_h_o_gen_list, b_h_gen_list, b_o_gen_list






ants = np.zeros([4, ants_per_generation, generations])
rewards = np.zeros([ants_per_generation, generations])

w_input_hidden_list = np.zeros([2,4,ants_per_generation])
w_hidden_output_list = np.zeros([4,3,ants_per_generation])

hidden_bias_list = np.zeros([4,ants_per_generation])
output_bias_list = np.zeros([3,ants_per_generation])

direction_list = [[0,1],[1,0],[0,-1],[-1,0]]
best_reward = []


food = generate_food(terrain_size)
ant_position = np.random.randint(1,terrain_size-1,2)

#first generation
print('Generation 0')
for k in range(ants_per_generation):
    
    ant_direction  = np.array(direction_list[np.random.randint(4)])
    ant = np.append(ant_position, ant_direction)
     
    ants[:,k,0] = ant
    
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.scatter([ant_position[0],food[0]], [ant_position[1],food[1]], color=['r','b'])
    # plt.xlim([0,100])
    # plt.ylim([0,100])  
       
    
    w_input_hidden = np.random.rand(2,4)
    w_hidden_output = np.random.rand(4,3)
    
    b_hidden = np.random.rand(4,)
    b_output = np.random.rand(3,)
    
    w_input_hidden_list[:,:,k] = w_input_hidden
    w_hidden_output_list[:,:,k] = w_hidden_output
    hidden_bias_list[:,k] = b_hidden
    output_bias_list[:,k] = b_output
     
    new_ant, reward, food = running_ant(ant, food, w_input_hidden, w_hidden_output,
                      b_hidden, b_output ,steps_per_run, terrain_size)
    
    starting_food_distance = np.linalg.norm(ant[:2]-food) if \
        np.linalg.norm(ant[:2]-food) > 0 else terrain_size*math.sqrt(2)
    final_food_distance = np.linalg.norm(new_ant[:2]-food)
    reward_distance = (starting_food_distance - final_food_distance)/starting_food_distance
    reward = reward + reward_distance
    rewards[k,0] = reward

print(rewards[:,0][np.argmax(rewards[:,0])])

# draw best ant 
gen = 0

index_reward_max = np.argmax(rewards[:,gen])

w_in_h = w_input_hidden_list[:,:,index_reward_max]
w_h_o = w_hidden_output_list[:,:,index_reward_max]
b_h = hidden_bias_list[:,index_reward_max]
b_o = output_bias_list[:,index_reward_max]

save_weighs_and_biases(w_input_hidden_list, w_hidden_output_list, 
                       hidden_bias_list, output_bias_list, gen)

if draw_iterations:
    np.save(directory + weights_and_biases_folder + '{:03d}'.format(run) + 
            '\\' + 'Rewards', rewards)

if draw_iterations:
    see_run(steps_per_run, terrain_size, gen)





#other generations

for gen in range(1,generations):
    print('Generation ' + str(gen))
    index_reward_max = np.argmax(rewards[:,gen-1])
    best_reward.append(rewards[:,gen-1][np.argmax(rewards[:,gen-1])])
    
    rewards[:,gen-1][np.argmax(rewards[:,gen-1])]
    
    food = generate_food(terrain_size)
    ant_position = np.random.randint(1,terrain_size-1,2)
    
    
    w_input_hidden_list[:,:,:], w_hidden_output_list[:,:,:], \
    hidden_bias_list[:,:], output_bias_list[:,:] = \
    generate_population(w_input_hidden_list, w_hidden_output_list, 
                        hidden_bias_list, output_bias_list, elitism, 
                        gen-1, mutation_rate)
    
    save_weighs_and_biases(w_input_hidden_list, w_hidden_output_list, 
                           hidden_bias_list, output_bias_list, gen)
    
    for k in range(ants_per_generation):

        ant_direction  = np.array(direction_list[np.random.randint(4)])
        ant = np.append(ant_position, ant_direction)
         
        ants[:,k,gen] = ant
        
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.scatter([ant_position[0],food[0]], [ant_position[1],food[1]], color=['r','b'])
        # plt.xlim([0,100])
        # plt.ylim([0,100])  
        
        w_input_hidden = w_input_hidden_list[:,:, k]
        w_hidden_output = w_hidden_output_list[:,:, k]
        b_hidden = hidden_bias_list[:, k]
        b_output = output_bias_list[:, k]
        
        
        new_ant, reward, food = running_ant(ant, food, w_input_hidden, 
                                        w_hidden_output, b_hidden, b_output, 
                                        steps_per_run, terrain_size)
        
        starting_food_distance = np.linalg.norm(ant[:2]-food) if \
            np.linalg.norm(ant[:2]-food) > 0 else terrain_size*math.sqrt(2)
        final_food_distance = np.linalg.norm(new_ant[:2]-food)
        reward_distance = (starting_food_distance - final_food_distance)/starting_food_distance
        reward = reward + reward_distance
        rewards[k,gen] = reward
    
    
    print(rewards[:,gen][np.argmax(rewards[:,gen])])
    
    # run best ant 
    index_reward_max = np.argmax(rewards[:,gen])

    w_in_h = w_input_hidden_list[:,:,index_reward_max]
    w_h_o = w_hidden_output_list[:,:,index_reward_max]
    b_h = hidden_bias_list[:,index_reward_max]
    b_o = output_bias_list[:,index_reward_max]
    
    if draw_iterations:
        np.save(directory + weights_and_biases_folder + '{:03d}'.format(run) + 
                '\\' + 'Rewards', rewards)
    if draw_iterations and gen%draw_every_n_iterations == 0:
        see_run(steps_per_run, terrain_size, gen)
        
best_reward.append(rewards[:,gen][np.argmax(rewards[:,gen])])

np.save(directory + weights_and_biases_folder + '{:03d}'.format(run) + 
        '\\' + 'Rewards', rewards)

# run best ant of gen
# food = generate_food(terrain_size)
# rewards = load_rewards(run)
# index_reward_max = np.argmax(rewards[:,gen])

# ant_position = np.random.randint(0,terrain_size,2)
# ant_direction  = np.array(direction_list[np.random.randint(4)])
# ant = np.append(ant_position, ant_direction)

# w_input_hidden_list, w_hidden_output_list, hidden_bias_list, \
#     output_bias_list =  load_weights_and_biases(gen)

# w_in_h = w_input_hidden_list[:,:,index_reward_max]
# w_h_o = w_hidden_output_list[:,:,index_reward_max]
# b_h = hidden_bias_list[:,index_reward_max]
# b_o = output_bias_list[:,index_reward_max]

see_run(steps_per_run, terrain_size, gen)

plt.plot(list(range(generations)), best_reward)
plt.ylim([0,rewards[:, :].max()*1.2])
plt.grid(visible=True)
plt.title('Rewards of best of each generation for run '+ str(run))
plt.xlabel('Generation')
plt.ylabel('Reward')
plt.savefig(os.path.join(directory, fig_folder, 'fig' + '{:03d}'.format(run)))
















