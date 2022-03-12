import torch,random
import argparse
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys,time
from scipy.special import softmax
sys.path.append("../")
from Methods.model_pretrain import TimeSlicePre
from Methods.utils import data_get_retweets



parser = argparse.ArgumentParser(description='Sequence Modeling - The Popularity Prediction Problem')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: True)')
parser.add_argument('--cuda_index', type=int, default=0,
                    help='index of cuda (default: 0)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=8,
                    help='kernel size (default: 8)')
parser.add_argument('--mlp_hid_size', type=int, default=8,
                    help='hidden size in mlp (default: 8)')

parser.add_argument('--levels', type=int, default=9,
                    help='# of levels (default: 9)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')


parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight of l2 regularizations(default: 1e-5)')

parser.add_argument('--max_label', type=int, default=48,
                    help='hidden size in mlp (default: 48)')
parser.add_argument('--time_slices', type=int, default=1800,
                    help='hidden size in mlp (default: 1800)')


parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=8,
                    help='number of hidden units per layer (default: 8)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--max_try', type=int, default=20,
                    help='number of max try (default: 20')

parser.add_argument('--interval_time', type=int, default=5,
                    help='time interval of inputs (default: 5')

parser.add_argument('--file_dir', type=str, default='../data/twitter/',
                    help='the directory of the data file (default: ../data/twitter/)')

parser.add_argument('--filename_train', type=str, default='pretrain_data_train.txt',
                    help='filename of data (default:pretrain_data_train.txt)')

parser.add_argument('--filename_val', type=str, default='pretrain_data_val.txt',
                    help='filename of data (default:pretrain_data_val.txt)')

parser.add_argument('--dataset', type=str, default='twitter',
                    help='the name of datasets: weibo or twitter (default: twitter)')


args = parser.parse_args()
random.seed(0)
np.random.seed(0)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

input_channels = 1
n_classes = 1
batch_size = args.batch_size
epochs = args.epochs
max_try = args.max_try

print(args)
print("Producing data...")

start_time = time.time()
retweets_train = data_get_retweets(args.file_dir+args.filename_train, min_time=2*args.time_slices)
retweets_val = data_get_retweets(filename=args.file_dir+args.filename_val, min_time=2*args.time_slices)
print("Finish Producing data! Total time:",time.time()-start_time)

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
mlp_hid_size = args.mlp_hid_size
dropout = args.dropout
device = torch.device("cpu")
if args.cuda:
    device = torch.device("cuda:" + str(args.cuda_index))
print("device:",device)
model = TimeSlicePre(input_channels, n_classes, channel_sizes,kernel_size=kernel_size,mlp_hid_size = mlp_hid_size, dropout=dropout)

if args.cuda:
    model.to(device)

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.weight_decay)
print("model paramerters",[x.numel() for x in model.parameters()])
print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
def get_batch(i,batch_size,retweets,args,if_print = False):
    if i + batch_size > len(retweets):
        batch_retweets = retweets[i:]
    else:
        batch_retweets = retweets[i:(i + batch_size)]
    X_A = np.zeros([len(batch_retweets),1,int(args.time_slices/args.interval_time)])
    X_B = np.zeros([len(batch_retweets),1,int(args.time_slices/args.interval_time)])
    Y = []
    for idx_batch in range(len(batch_retweets)):
        temp_retweets = batch_retweets[idx_batch]
        temp_idx = int(len(temp_retweets)*0.9)
        last_time_slice = int(temp_retweets[temp_idx]/args.time_slices)
        label_range = min(args.max_label,last_time_slice)
        
        
        slice_ok = False
        n_count = 0
        while  not(slice_ok):
            generate_label = random.randint(1,label_range-1)
        
            A_slice_choice = np.arange(0,last_time_slice-generate_label)
            prob_choice = softmax(np.log(1.0/np.power(A_slice_choice+1,2)))
            time_slice_A = np.random.choice(A_slice_choice,p = prob_choice)
            time_slice_B = time_slice_A + generate_label
        
            #print("generate label, time slice A and B:",generate_label,time_slice_A,time_slice_B)
            n_A = 0
            n_B = 0
            for t in temp_retweets:
                if t >= time_slice_A * args.time_slices and t < (time_slice_A+1)*args.time_slices:
                    n_A +=1
                if t >= time_slice_B * args.time_slices and t < (time_slice_B+1)*args.time_slices:
                    n_B +=1
            if n_A >0 and n_B >0:
                slice_ok = True
                #print("slice ok!") 
            n_count +=1
            if n_count >10:
                slice_ok = True
                #print("too many samples! slice ok")
        #if if_print:
        #    print("last time slice, label range,generate label, time slice A and B:",
        #                        last_time_slice, label_range, generate_label,time_slice_A,time_slice_B) 
        for t in temp_retweets:
            if t >= time_slice_A * args.time_slices and t < (time_slice_A+1)*args.time_slices:
                t_index = int((t-time_slice_A * args.time_slices)*1.0/args.interval_time)
                X_A[idx_batch,0,t_index] +=1
            if t >= time_slice_B * args.time_slices and t < (time_slice_B+1)*args.time_slices:
                t_index = int((t-time_slice_B * args.time_slices)*1.0/args.interval_time)
                X_B[idx_batch,0,t_index] +=1

        Y.append([generate_label])
        #print("X_A:",X_A[idx_batch])
        #print("X_B:",X_B[idx_batch])
        #print("Y:",Y[idx_batch])        


    torch_X_A = torch.tensor(X_A,dtype = torch.float32,requires_grad=False)
    torch_X_B = torch.tensor(X_B,dtype = torch.float32,requires_grad=False)
    torch_Y = torch.tensor(Y,dtype=torch.float32, requires_grad=False)

    if args.cuda:
        device = torch.device("cuda:" + str(args.cuda_index))
        torch_X_A = torch_X_A.to(device)
        torch_X_B = torch_X_B.to(device)
        torch_Y = torch_Y.to(device)
    return torch_X_A, torch_X_B,torch_Y

def evaluate(X1,X2,Y,model):
    model.eval()
    with torch.no_grad():
        output = model(X1,X2)
        test_loss = F.mse_loss(output, Y)
        #print("true:",Y)
        #print("pred:",output)
        return test_loss.item(),output


Best_val_loss = 1000000000.0
Patience = max_try
start_time = time.time()
for epoch in range(1, epochs + 1):
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, len(retweets_train), batch_size):
        x1, x2, y = get_batch(i,batch_size,retweets_train,args)
        optimizer.zero_grad()
        output = model(x1,x2)
        loss = F.mse_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        batch_idx +=1
        
        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, len(retweets_train))

            val_loss_list = []
            for jj in range(0,100):
                j = random.randrange(0,len(retweets_val),batch_size)
                if jj == 0:
                    x1_val, x2_val, y_val = get_batch(j, batch_size, retweets_val,args,False)
                else:
                    x1_val, x2_val, y_val = get_batch(j, batch_size, retweets_val,args)
                temp_loss, predy_val = evaluate(x1_val,x2_val,y_val,model)
                val_loss_list.append(temp_loss)
            val_loss = np.mean(val_loss_list)

            if val_loss < Best_val_loss:
                Patience = max_try
                Best_val_loss = val_loss
                torch.save(model,'./save_models/pretrain_'+str(args.dataset)+'_model_max_label'+str(args.max_label)+"_time_slices"+str(args.time_slices)+"_lr"+str(args.lr)+"_wd"+str(args.weight_decay)+'.pkl')

            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tTrain Loss: {:.6f}\tVal Loss: {:.6f}'.format(
                epoch, processed, len(retweets_train), 100.*processed/len(retweets_train), cur_loss,val_loss))
            print('Best val loss: {:.6f}\tPatience: {:2d}\tTime: {:.6f}'.format(
                Best_val_loss,Patience,time.time()-start_time))

            total_loss = 0
            Patience  = Patience -1
            if Patience <0 :
                test_loss = 0
                print('Best val loss: {:.6f}\tBest test loss: {:.6f}\tTotal Time: {:.6f}'.format(
                    Best_val_loss, test_loss,time.time()-start_time))
                break
        if Patience <0:
            break
    if Patience <0:
        break
print('Best val loss: {:.6f}\tBest test loss: {:.6f}\tTotal Time: {:.6f}'.format(
    Best_val_loss, test_loss, time.time() - start_time))


