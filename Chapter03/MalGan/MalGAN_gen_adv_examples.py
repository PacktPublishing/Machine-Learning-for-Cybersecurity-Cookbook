from sklearn.neighbors import NearestNeighbors
from keras import backend as K
import MalGAN_utils
from MalGAN_preprocess import preprocess
import numpy as np

def gen_adv_samples(model, fn_list, pad_percent=0.1, step_size=0.001, thres=0.5):

    ###   search for nearest neighbor in embedding space ###
    def emb_search(org, adv, pad_idx, pad_len, neigh):
        out = org.copy()
        for idx in range(pad_idx, pad_idx+pad_len):
            target = adv[idx].reshape(1, -1)
            best_idx = neigh.kneighbors(target, 1, False)[0][0]
            out[0][idx] = best_idx
        return out


    max_len = int(model.input.shape[1])
    emb_layer = model.layers[1]
    emb_weight = emb_layer.get_weights()[0]
    inp2emb = K.function([model.input]+ [K.learning_phase()], [emb_layer.output]) # [function] Map sequence to embedding

    # Build neighbor searches
    neigh = NearestNeighbors(1)
    neigh.fit(emb_weight)

    log = MalGAN_utils.logger()
    adv_samples = []

    for e, fn in enumerate(fn_list):

        ###   run one file at a time due to different padding length, [slow]
        inp, len_list = preprocess([fn], max_len)
        inp_emb = np.squeeze(np.array(inp2emb([inp, False])), 0)

        pad_idx = len_list[0]
        pad_len = max(min(int(len_list[0]*pad_percent), max_len-pad_idx), 0)
        org_score = model.predict(inp)[0][0]    ### origianl score, 0 -> malicious, 1 -> benign
        loss, pred = float('nan'), float('nan')

        if pad_len > 0:

            if org_score < thres:
                adv_emb, gradient, loss = fgsm(model, inp_emb, pad_idx, pad_len, e, step_size)
                adv = emb_search(inp, adv_emb[0], pad_idx, pad_len, neigh)
                pred = model.predict(adv)[0][0]
                final_adv = adv[0][:pad_idx+pad_len]

            else: # use origin file
                final_adv = inp[0][:pad_idx]


        log.write(fn, org_score, pad_idx, pad_len, loss, pred)

        # sequence to bytes
        bin_adv = bytes(list(final_adv))
        adv_samples.append(bin_adv)

    return adv_samples, log