from collections import OrderedDict
import os
import sys
basedir = os.getcwd()
sys.path.append(basedir)
from multihead_attention_model_torch import *
from Genedata import Gene_data
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
import re
import pickle
from utils import *


device = "cuda"


encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])

encoding_fold = {'(': [1, 0, 0, 0],
                ')': [0, 1, 0, 0],
                '.': [0, 0, 1, 0],
                'N': [0, 0, 0, 1]}

seq_encoding_keys = list(encoding_seq.keys())
# global seq_encoding_keys
fold_encoding_keys = list(encoding_fold.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))

gene_ids = None

def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
    return weights





def typeicalSampling(ids, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=1234)
    folds = kf.split(ids)
    train_fold_ids = OrderedDict()
    val_fold_ids = OrderedDict()
    test_fold_ids=OrderedDict()
    for i, (train_indices, test_indices) in enumerate(folds):
        size_all = len(train_indices)
        train_fold_ids[i] = []
        val_fold_ids[i] = []
        test_fold_ids[i]  =[]
        train_indices2 = train_indices[:int(size_all * 0.8)]
        val_indices = train_indices[int(size_all * 0.8):]
        for s in train_indices2:
             train_fold_ids[i].append(ids[s])
        
        for s in val_indices:
             val_fold_ids[i].append(ids[s])
        
        for s in test_indices:
              test_fold_ids[i].append(ids[s])
        
    
    return train_fold_ids,val_fold_ids,test_fold_ids

def group_sample(label_id_Dict,datasetfolder, dataset_name,foldnum=8):
    Train = OrderedDict()
    Test = OrderedDict()
    Val = OrderedDict()
    for i in range(foldnum):
        Train.setdefault(i,list())
        Test.setdefault(i,list())
        Val.setdefault(i,list())
    
    for eachkey in label_id_Dict:#decide which item was used to split k fold
        label_ids = list(label_id_Dict[eachkey])
        if len(label_ids)<foldnum:
            for i in range(foldnum):
                Train[i].extend(label_ids)
            
            continue
        
        [train_fold_ids, val_fold_ids,test_fold_ids] = typeicalSampling(label_ids, foldnum)
        for i in range(foldnum):
            Train[i].extend(train_fold_ids[i])
            Val[i].extend(val_fold_ids[i])
            Test[i].extend(test_fold_ids[i])
            # print('label:%s finished sampling! Train length: %s, Test length: %s, Val length:%s'%(eachkey, len(train_fold_ids[i]), len(test_fold_ids[i]),len(val_fold_ids[i])))
    
    for i in range(foldnum):
        # print("spliting the data into:%s folds" % foldnum)
        # print('Train length: %s, Test length: %s, Val length: %s'%(len(Train[i]),len(Test[i]),len(Val[i])))
        #print(type(Train[i]))
        #print(Train[0][:foldnum])
        #save in pickle
        pickle.dump(Train[i], open(datasetfolder+ "/" + dataset_name +'/Train' + str(foldnum) +str(i)+'.pkl',"wb"))
        pickle.dump(Test[i], open(datasetfolder+ "/" + dataset_name +'/Test' + str(foldnum) +str(i)+'.pkl',"wb"))
        pickle.dump(Val[i], open(datasetfolder+ "/" + dataset_name +'/Val' + str(foldnum) +str(i)+'.pkl',"wb"))
        np.savetxt(datasetfolder+ "/" + dataset_name +'/Train' + str(foldnum) +str(i)+'.txt', np.asarray(Train[i]),fmt="%s")
        np.savetxt(datasetfolder+ "/" + dataset_name + '/Test' + str(foldnum) +str(i)+'.txt', np.asarray(Test[i]),fmt="%s")
        np.savetxt(datasetfolder+ "/" + dataset_name +'/Val' + str(foldnum) +str(i)+'.txt', np.asarray(Val[i]),fmt="%s")
    
    return Train, Test, Val



def maxpooling_mask(input_mask,pool_length=3):
    #input_mask is [N,length]
    max_index = int(input_mask.shape[1]/pool_length)-1
    max_all=np.zeros([input_mask.shape[0],int(input_mask.shape[1]/pool_length)])
    for i in range(len(input_mask)):
        index=0
        for j in range(0,len(input_mask[i]),pool_length):
            if index<=max_index:
                max_all[i,index] = np.max(input_mask[i,j:(j+pool_length)])
                index+=1
    
    return max_all





def preprocess_data(left=4000, right=4000, dataset='/home/sxr280/DeepRBPLoc/testdata/modified_multilabel_seq_nonredundent.fasta',padmod='center',pooling_size=8, foldnum=4, pooling=True, RNA_type = None, RNA_tag = False):
    # print("loading the gene object")
    gene_data = Gene_data.load_sequence(dataset, left, right, RNA_type=RNA_type)
    # print("after loading the gene object")
    id_label_seq_Dict = get_id_label_seq_Dict(gene_data)
    label_id_Dict = get_label_id_Dict(id_label_seq_Dict)
    Train=OrderedDict()
    Test=OrderedDict()
    Val=OrderedDict()
    datasetfolder=os.path.dirname(dataset)
    dataset_name = dataset.split("/")[-1].split(".")[0]
    if os.path.exists(datasetfolder+ "/" + dataset_name+'/Train5'+str(0)+'.pkl'):
        for i in range(5):
            Train[i] = pickle.load(open(datasetfolder + "/" + dataset_name +'/Train5'+str(i)+'.pkl', "rb"))
            Test[i] = pickle.load(open(datasetfolder + "/" + dataset_name +'/Test5'+str(i)+'.pkl', "rb"))
            Val[i] = pickle.load(open(datasetfolder + "/" + dataset_name +'/Val5'+str(i)+'.pkl', "rb"))

            # Train[i] = np.loadtxt(datasetfolder + "/" + dataset_name +'/Train5'+str(i)+'.txt',dtype='str', delimiter = "\n")#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Train')[:]
            # Test[i] = np.loadtxt(datasetfolder + "/" + dataset_name +'/Test5'+str(i)+'.txt',dtype='str', delimiter = "\n")#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Test')[:]
            # Val[i] = np.loadtxt(datasetfolder + "/" + dataset_name + '/Val5'+str(i)+'.txt',dtype='str', delimiter = "\n")#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Val')[:]
    else:

        # print("creating group sampling")
        [Train, Test,Val] = group_sample(label_id_Dict,datasetfolder, dataset_name,foldnum)
    
    Xtrain={}
    Xtest={}
    Xval={}
    Ytrain={}
    Ytest={}
    Yval={}
    Train_mask_label={}
    Test_mask_label={}
    Val_mask_label={}
    maxpoolingmax = int((left+right)/pooling_size)
    global seq_encoding_keys
    if RNA_type == "allRNA":
        with open(dataset, "r") as f1:
            string = f1.read()
        pattern = r"RNA_category:([^,\n]+)"
        RNA_types = re.findall(pattern, string)
        RNA_types = list(sorted(set(RNA_types)))
        # print("RNA_types", RNA_types)
        seq_encoding_keys += RNA_types
        encoding_keys = seq_encoding_keys
        # print("new encoding list:", encoding_keys)
    else:
        encoding_keys = seq_encoding_keys
    for i in range(foldnum):
        #if i <2:
        #   continue
        
        # print('padding and indexing data')
        
        # fold_encoding_keys  = fold_encoding_keys 
        encoding_vectors = seq_encoding_vectors
        #train
        #padd center
        # print("Train[i]:", Train[i])
        X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Train[i]]
        X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Train[i]]
        
        # print("original seq:", X_left)


        if padmod =='center':
            mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
            mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
            mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
            Train_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            X_left = tf.keras.utils.pad_sequences(X_left,maxlen=left,
                               dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')  #padding after sequence
            
            X_right = tf.keras.utils.pad_sequences(X_right,maxlen=right,
                          dtype=np.int8, value=encoding_keys.index('UNK'),padding='pre')# padding before sequence
            
            Xtrain[i] = np.concatenate([X_left,X_right],axis = -1)
        else:
           #merge left and right and padding after sequence
           Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
           #adding additional tag for each sequence
           # print("Xall before:", np.array(Xall).shape, "example:", Xall[0])
           if RNA_tag:
               Xall = get_new_seq(Train[i], Xall, encoding_keys, left, right)
           # print("Xall shape:", np.array(Xall).shape, Xall[0])
        #    print("where < 6 :", len(Xall), Xall)
        #    mRNA_site = Xall
           # print("before padding:", Xall[12])
           Xtrain[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
        #    Xtrain[i] = np.zeros(left+right，)
           # print("after padding:", Xtrain[i][12])
        #    for i in Xtrain[i]:
        #        tag = i[0]
        #        if tag < 6:
        #            print(i)
           if pooling == False:
               maxpoolingmax=8000
               Train_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene))),np.zeros(maxpoolingmax-int(len(gene)))]) for gene in Xall],dtype='float32')

           else:
               Train_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')

               

        Ytrain[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Train[i]])
        # print("training shapes"+str(Xtrain[i].shape)+" "+str(Ytrain[i].shape))
        
        #test
        X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Test[i]]
        X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Test[i]]
       
        if padmod =='center':
            mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
            mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
            mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
            Test_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            X_left = pad_sequences(X_left,maxlen=left,
                               dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')  #padding after sequence
            
            X_right = pad_sequences(X_right,maxlen=right,
                          dtype=np.int8, value=encoding_keys.index('UNK'),padding='pre')# padding before sequence
            
            Xtest[i] = np.concatenate([X_left,X_right],axis = -1)
        else:
            #merge left and right and padding after sequence
            Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
            if RNA_tag:
               Xall = get_new_seq(Test[i], Xall, encoding_keys, left, right)
            # mRNA_site = np.where(Xall[:,0] == 9)[0]
            # print("number of mRNA in test fold", i, "is:", len(mRNA_site))
            Xtest[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
            
            if pooling == False:
               maxpoolingmax=8000
               Test_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene))),np.zeros(maxpoolingmax-int(len(gene)))]) for gene in Xall],dtype='float32')
            else:
               Test_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')



        Ytest[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Test[i]])
        #validation
        X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Val[i]]
        X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Val[i]]
        
        if padmod=='center':
            mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
            mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
            mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
            Val_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            X_left = pad_sequences(X_left,maxlen=left,
                               dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')  #padding after sequence
            
            X_right = pad_sequences(X_right,maxlen=right,
                          dtype=np.int8, value=encoding_keys.index('UNK'),padding='pre')# padding before sequence
            
            Xval[i] = np.concatenate([X_left,X_right],axis = -1)
        else:
            #merge left and right and padding after sequence
            Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
            if RNA_tag:
               Xall = get_new_seq(Val[i], Xall, encoding_keys, left, right)
            Xval[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
            
            if pooling == False:
               maxpoolingmax=8000
               Val_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene))),np.zeros(maxpoolingmax-int(len(gene)))]) for gene in Xall],dtype='float32')
            else:
               Val_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')

        Yval[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Val[i]])
    return Xtrain,Ytrain,Train_mask_label,Xtest,Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors


def preprocess_data2(left=4000, right=4000, dataset='/home/sxr280/DeepRBPLoc/testdata/modified_multilabel_seq_nonredundent.fasta',padmod='center',pooling_size=8, foldnum=4, pooling=True, RNA_type = None, RNA_tag = False):
    '''
    This is not slit version
    '''
    # print("loading the gene object")
    gene_data = Gene_data.load_sequence(dataset, left, right, RNA_type=RNA_type)
    # print("after loading the gene object")
    id_label_seq_Dict = get_id_label_seq_Dict(gene_data)

    maxpoolingmax = int((left+right)/pooling_size)
    global seq_encoding_keys
    if RNA_type == "allRNA":
        root_dir=os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        
        overall_dataset = pj(root_dir, "data", "allRNA", "allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta")
        with open(overall_dataset, "r") as f1:
            string = f1.read()
        pattern = r"RNA_category:([^,\n]+)"
        RNA_types = re.findall(pattern, string)
        RNA_types = list(sorted(set(RNA_types)))
        # print("RNA_types", RNA_types)
        seq_encoding_keys += RNA_types
        encoding_keys = seq_encoding_keys
        # print("new encoding list:", encoding_keys)
    else:
        encoding_keys = seq_encoding_keys
    
        
    # print('padding and indexing data')
        
    encoding_vectors = seq_encoding_vectors
    X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in id_label_seq_Dict.keys()]
    X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in id_label_seq_Dict.keys()]


    #merge left and right and padding after sequence
    Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
    #adding additional tag for each sequence
    # print("Xall before:", np.array(Xall).shape, "example:", Xall[0])
    if RNA_tag:
        Xall = get_new_seq(list(id_label_seq_Dict.keys()), Xall, encoding_keys, left, right)
    # print("Xall shape:", np.array(Xall).shape, Xall[0])

    # print("before padding:", Xall[12])
    X = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')

    if pooling == False:
        maxpoolingmax=8000
        mask_label=np.array([np.concatenate([np.ones(int(len(gene))),np.zeros(maxpoolingmax-int(len(gene)))]) for gene in Xall],dtype='float32')

    else:
        mask_label=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')
    ids = list(id_label_seq_Dict.keys())
        
      
    return X,mask_label,ids


def gin_file_to_dict(file_path):
    with open(file_path, 'r') as f:
        gin_config = f.read()
    # Extract configuration lines from the gin file
    config_lines = re.findall(r'^\w[\w\.]+ = .+', gin_config, re.MULTILINE)
    # Convert the configuration lines into a dictionary
    config_dict = {}
    for line in config_lines:
        key, value = line.split(' = ')
        config_dict[key] = value
    return config_dict










