import torch, random
import numpy as np
import pickle as pkl
import datetime

def data_get_retweets(filename,min_time):
    all_retweets = []
    file = open(filename,"r",encoding="utf8")
    for line in file:
        retweets = []
        parts = line.split("\t")
        records = parts[6].split(" ")
        for r in records:
            time = int(r.split(":")[1])
            retweets.append(time)
        retweets.sort()
        #print(retweets)
        temp_idx = int(len(retweets) * 0.9)
        if retweets[temp_idx] < min_time:
            continue
        all_retweets.append(retweets)
    print("data size:",len(all_retweets)) 
    return all_retweets

def feature_generator_num(filename, ob_time, gap_time,drop_prob=0.0,reg = 1, label_time = 30*3600*24):
    """
    Args:
        filename: the name of the data file, obeys the format
        ob_time: the time of observations
        gap_time: the time gap of each tcn input
    """
    random.seed(0)
    data_X = []
    data_Y = []

    file = open(filename,"r",encoding="utf8")
    n_line = 0
    n_ok_line = 0
    for line in file:
        n_line +=1
        parts = line.split("\t")
        if len(parts) != 7:
            continue

        x = np.zeros(int(ob_time/gap_time))

        retweets = parts[6].split(" ")
        y_within_label_time = 0
        for r in retweets:
            t = int(r.split(":")[1])
            if t < ob_time:
                index = int(t*1.0/gap_time)
                x[index] +=1
                #x.append(index)
            if t < label_time:
                y_within_label_time +=1       
 
        #pubtime = int(parts[3])
        #date_time = datetime.datetime.fromtimestamp(pubtime)
        #hour = date_time.hour

        #y = np.log(int(parts[5])-len(x)+1.0) / np.log(2.0)
        # Mean Log-transformed Square Error Label
        #y = np.log(int(parts[5])+1.0) / np.log(2.0)
        # Mean Relative Square Error Label
        
        if reg == 1:
            y = y_within_label_time
        elif reg == 0:
            if y_within_label_time >= 2*np.sum(x):
                y = 1
            else:
                y = 0
        else:
            print("error! 'reg' should be 0 or 1.") 
        #y = int(parts[5])   
        #if int(parts[5]) != y:
        #    print("not equal!",int(parts[5]),y)
        if int(parts[5])-np.sum(x)+1.0 <0:
            print("error!",int(parts[5])-len(x)+1.0)
        # np.log(y + 1.0) / np.log(2.0)
        #y = np.log(int(parts[5]))
        #if len(x) < 10 or hour <8 or hour >18 or len(x) >1000 :
        #    continue
        r = random.random()
        if r < drop_prob:
            continue
        data_X.append(x)
        data_Y.append([y])
#        if reg == 1:
#            data_Y.append([y])
#        elif reg == 0:
#            data_Y.append(y)
#        else:
#            print("error! 'reg' should be 0 or 1")
        n_ok_line +=1
    print("number of lines:",n_line," number of ok lines",n_ok_line)
    print("data size:",len(data_X),len(data_Y))
    print("label percentage:",np.sum(data_Y),len(data_Y),np.sum(data_Y)*1.0/len(data_Y))

    return data_X,data_Y

def data_generator_num(filename, ob_time, gap_time,drop_prob=0.0,reg = 1, label_time = 30*3600*24):
    """
    Args:
        filename: the name of the data file, obeys the format
        ob_time: the time of observations
        gap_time: the time gap of each tcn input
        reg: 1 indicate the regression task (the value of popularity), 0 indicate the classification task (whetherr the final popularity is double increased)
        label_time: the horizon time (days) for predicted popularity. 
    """
    random.seed(0)
    data_X = []
    data_Y = []

    file = open(filename,"r",encoding="utf8")
    n_line = 0
    n_ok_line = 0
    for line in file:
        n_line +=1
        parts = line.split("\t")
        if len(parts) != 7:
            continue

        x = []
        y_within_label_time = 0

        retweets = parts[6].split(" ")
        for r in retweets:
            t = int(r.split(":")[1])
            if t < ob_time:
                index = int(t*1.0/gap_time)
                x.append(index)
            if t < label_time:
                y_within_label_time +=1
        
        if reg == 1:
            y = y_within_label_time
        elif reg == 0:
            if y_within_label_time >= 2*len(x):
                y = 1
            else:
                y = 0
        else:
            print("error! 'reg' should be 0 or 1.") 
        if int(parts[5])-len(x)+1.0 <0:
            print("error!",int(parts[5])-len(x)+1.0)
        r = random.random()
        if r < drop_prob:
            continue
        data_X.append(x)
        data_Y.append([y])
        n_ok_line +=1
        if n_ok_line % 10000 ==0:
            print(np.sum(data_Y),len(data_Y))
    print("number of lines:",n_line," number of ok lines",n_ok_line)
    print("data size:",len(data_X))
    print("label percentage:",np.sum(data_Y),len(data_Y),np.sum(data_Y)*1.0/len(data_Y))

    return data_X,data_Y

def data_loader(filein_dir,ob_time,gap_time,pop_threshold,
                       split="random"):
    """
    Args:
        filename: the name of the data file, obeys the format
        ob_time: the time of observations
        split: the way of split, choose from "random" and "pubtime"

    """
    print("start data load!")
    [train_X, train_Y, val_X, val_Y, test_X, test_Y] = \
        pkl.load(open(filein_dir+"dataset_obtime"+str(ob_time)
                  +"_gaptime"+str(gap_time)+"_threshol"+str(pop_threshold)
                  +"_split"+split,"rb"))

if __name__ == "__main__":
    #data_loader(filein_dir = "../../../data/weibo/",
    #                    ob_time = 3600,gap_time = 5,pop_threshold = 10,
    #                    split="random")

    data_generator_num(filename = "../data/twitter/downstream_data_train.txt", 
                        ob_time = 1800, gap_time = 5, drop_prob=0.0,reg = 0, label_time = 7*3600*24)

    #feature_generator_num(filename = "../data/weibo_large/cascades_train_20160601to0609_10_contextnew.txt", 
    #                    ob_time = 3600, gap_time = 600, drop_prob=0.0,reg = 1, label_time = 20*3600*24)

