#!/usr/bin/python
import re
import sys
import numpy as np


def dico(emb):
    f = open(emb, "r")
    lines = f.readlines()
    dico = dict() #word -> id
    for line in lines:
        arg = line.split()
        dico[arg[0]] = np.array(arg[1:])
    f.close()
    return dico


def conll_parser(filename):
    f = open(filename, 'r')
    all_phrase = f.readlines()
    f.close()
    phrase = []
    sep_phrase = []
    for line in all_phrase:
        if line != "\n" :
            phrase.append(line)
        else:
            sep_phrase.append(phrase)
            phrase = []
    surf_list = []
    deep_list = []
    sense = ""
    ele_copy = ""
    for each in sep_phrase :
        verb_ind = []
        for ele in each :
            ele = ele.split()
            if "sense=" in ele[5] :
                sense = ele[5].split("#")
                ele_copy = ele
                verb_ind.append(ele[0])
        for index in verb_ind :
            local_surf = []
            local_deep = []
            local_surf.append([ele_copy[1],sense[1]])
            local_deep.append([ele_copy[1],sense[1]])
            for ele in each :
                ele = ele.split()
                ind = ele[6].split("|")
                if index in ind :
                    if "P" not in ele[3]:
                        ind = ele[6].split("|")
                        dep = ele[7].split("|")
                        ind_dep = dict(zip(ind,dep))
                        fonctions = ind_dep[index].split(":")
                        if len(fonctions) == 3 :
                            surf = fonctions[1]
                            deep = fonctions[2]
                            if surf in ["obj","suj","a_obj"] :
                                arg_surf = [ele[1],ele[3],surf]
                                local_surf.append(arg_surf)
                            if deep in ["obj","suj","a_obj"] :
                                arg_deep = [ele[1],ele[3],deep]
                                local_deep.append(arg_deep)
                        elif len(fonctions) == 2 :
                            surf = fonctions[0]
                            deep = fonctions[1]
                            if surf in ["obj","suj","a_obj"] :
                                arg_surf = [ele[1],ele[3],surf]
                                local_surf.append(arg_surf)
                            if deep in ["obj","suj","a_obj"] :
                                arg_deep = [ele[1],ele[3],deep]
                                local_deep.append(arg_deep)
            surf_list.append(local_surf)
            deep_list.append(local_deep)
    return (surf_list,deep_list)


#output : a list of POS,prepare for one-hot vector construction
def pos_list(filename):
    f = open(filename, 'r')
    all_phrase = f.readlines()
    f.close()
    phrase = []
    sep_phrase = []
    list_pos = []
    for line in all_phrase:
        if line != "\n" :
            phrase.append(line)
        else:
            sep_phrase.append(phrase)
            phrase = []
    for sep in sep_phrase :
        for ele in sep :
            ele = ele.split()
            if ele[3] not in list_pos :
                list_pos.append(ele[3])
    return list_pos


#output: a vector of 339 dimensions for each node
def single_vec(info_list,pos_dico,vec_dico, random): #affection before execution of this function
    suj_vec = np.array([])
    suj_pos = np.array([])
    obj_vec = np.array([])
    obj_pos = np.array([])
    abo_vec = np.array([])
    abo_pos = np.array([])
    if len(info_list) > 1:
        for info in info_list[1:] :
            #print(info[0])
            if info[2] == "suj" :
                if info[0] in vec_dico :
                    suj_vec = vec_dico[info[0]]
                else:
                    suj_vec = random
                pos_position = pos_dico.index(info[1])
                suj_pos = np.zeros(len(pos_dico))
                suj_pos[pos_position] = 1
            elif info[2] == "obj" :
                if info[0] in vec_dico :
                    obj_vec = vec_dico[info[0]]
                else :
                    obj_vec = random
                pos_position = pos_dico.index(info[1])
                obj_pos = np.zeros(len(pos_dico))
                obj_pos[pos_position] = 1
            elif info[2] == "a_obj" :
                if info[0] in vec_dico :
                    abo_vec = vec_dico[info[0]]
                else :
                    abo_vec = random
                pos_position = pos_dico.index(info[1])
                abo_pos = np.zeros(len(pos_dico))
                abo_pos[pos_position] = 1
    if not suj_vec.tolist() :
        suj_vec = np.zeros(100)
        suj_pos = np.zeros(len(pos_dico))
    if not obj_vec.tolist() :
        obj_vec = np.zeros(100)
        obj_pos = np.zeros(len(pos_dico))
    if not abo_vec.tolist():
        abo_vec = np.zeros(100)
        abo_pos = np.zeros(len(pos_dico))
    the_vec = np.concatenate([suj_vec, obj_vec, abo_vec, suj_pos, obj_pos, abo_pos], axis=0)
    return the_vec


#input : a conll file, the sigma value, the dependency structure we usage
#output : the accuracy of our prediction
def propagation(file_conll,sigma,deep_or_surf):
    surf, deep = conll_parser(file_conll)
    vec_dico = dico("vecs100-linear-frwiki")
    pos_dico = pos_list(file_conll)
    random = np.random.rand(1,100)
    random = random[0]
    if deep_or_surf == "deep" :
        surf_copy = surf
        surf = deep
    surf_exemples = []
    class_list = []
    for every_v in surf :
        class_list.append(every_v[0][1])
        surf_exemples.append(single_vec(every_v,pos_dico,vec_dico,random))
    uniq = list(set(class_list))
    train_array = np.linspace(0,len(surf_exemples),num=30, dtype=int,endpoint=False)
    y_matrix = np.zeros((len(surf_exemples),len(uniq))) # array of indexes
    for id in np.nditer(train_array) :
        class_id = surf[id][0][1]
        indice = uniq.index(class_id)
        y_matrix[id] = np.zeros(len(uniq))
        y_matrix[id,indice] = 1
    t_matrix = np.zeros((len(surf_exemples), len(surf_exemples)))
    surf_key = np.asarray(surf_exemples, dtype=np.float32)
    j_sim = []
    x = float(sigma)
    #essentiel step to construct the T matrix
    for j in range(len(surf_exemples)):
        kj_sim = 0
        for k in range(len(surf_exemples)):
            kj_sim += np.exp(-np.linalg.norm(surf_key[k]-surf_key[j])**2/x**2)
        j_sim.append(kj_sim)
    j_sim_array = np.asarray(j_sim)
    for i in range(len(surf_exemples)):
        summ = 0
        for j in range(len(surf_exemples)):
            ij_sim = np.exp(-np.linalg.norm(surf_key[i]-surf_key[j])**2/x**2)
            if i != j :
                summ += ij_sim
            t_matrix[i,j] = ij_sim/j_sim_array[j]
        #print(summ) # can be used to check whether the sum of no reflexive similarities is equal to zero when sigma is too small
    new_y_matrix = np.copy(y_matrix)
    last_matrix = np.copy(y_matrix)
    while True:
        diff_abs = 0
        #propagation
        for index in range(len(last_matrix)) :
            if index not in train_array :
                for c in range(len(uniq)) :
                    rate = 0
                    for j in range(len(t_matrix)):
                        rate += t_matrix[index, j]*last_matrix[j,c]
                    new_y_matrix[index, c] = rate
        #normalise Y
        for index in range(len(new_y_matrix)):
            sum_row = sum(new_y_matrix[index])
            for c in range(len(uniq)):
                new_y_matrix[index, c] = round(new_y_matrix[index,c]/sum_row,4)
                diff_abs += abs(new_y_matrix[index, c ] - last_matrix[index, c])
        #condition d'arrêt
        if diff_abs < 0.1 :
            break;
        last_matrix = np.copy(new_y_matrix)
    l = 0
    sentence = 0
    base = 0
    # calculate precision
    for i in range(len(new_y_matrix)):
        sentence += 1
        if max(new_y_matrix[i])!= 0 and i not in train_array:
            ind = new_y_matrix[i].tolist().index(max(new_y_matrix[i]))
            c = uniq[ind]
        else :
            c = "-1"
        if  c == class_list[i]:
            l += 1
        if  i not in train_array and class_list[i] == "2": # you can use this to calculate MFS
            base += 1
    accuracy = "{:.2%}".format(l/(len(new_y_matrix)-30))
    base_line = "{:.2%}".format(base/(len(new_y_matrix)-30)) # calculate MFS
    print(accuracy)
    #print(base_line) #you can use this to print MFS



if __name__ == '__main__':
    propagation(sys.argv[1], sys.argv[2], sys.argv[3]) #input : a conll file, the sigma value, the dependency structure we usage
