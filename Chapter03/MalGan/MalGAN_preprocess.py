import os
import time
import pickle
import argparse
import pandas as pd
from keras.preprocessing.sequence import pad_sequences


parser = argparse.ArgumentParser(description='Malconv-keras classifier')
parser.add_argument('--max_len', type=int, default=200000)
parser.add_argument('--save_path', type=str, default='../saved/preprocess_data.pkl')
parser.add_argument('csv', type=str)

def preprocess(fn_list, max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    corpus = []
    for fn in fn_list:
        if not os.path.isfile(fn):
            print(fn, 'not exist')
        else:
            with open(fn, 'rb') as f:
                corpus.append(f.read())
    
    corpus = [[byte for byte in doc] for doc in corpus]
    len_list = [len(doc) for doc in corpus]
    seq = pad_sequences(corpus, maxlen=max_len, padding='post', truncating='post')
    return seq, len_list


if __name__ == '__main__':
    args = parser.parse_args()
        
    df = pd.read_csv(args.csv, header=None)
    fn_list = df[0].values
    
    print('Preprocessing ...... this may take a while ...')
    st = time.time()
    processed_data = preprocess(fn_list, args.max_len)[0]
    print('Finished ...... %d sec' % int(time.time()-st))
    
    with open(args.save_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print('Preprocessed data store in', args.save_path)

