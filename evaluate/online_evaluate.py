
import numpy as np
from utils.scoring import compute_multiclass_accuracy
from tqdm import tqdm 

class EvaluatePrequentialSlidingWindow:
    def __init__(self, pretrain_size, window_size, slide_factor, window_wait):
        self.pretrain_size = pretrain_size
        self.window_size = window_size
        self.slide_factor = slide_factor
        self.window_wait = window_wait
    
    def evaluate(self, stream, model):
        # Warmup
        window = stream.next_sample(self.pretrain_size)
        X, y = window[0], window[1]
        model.partial_fit(X,y,classes=[0,1,2,3,4,5])

        first_run = True
        acc_win = []

        total_samples = stream.n_remaining_samples()
        # total_sliding_samples = ((total_samples-self.window_size)+1)*self.window_size
        total_sliding_samples = (((total_samples-self.window_size)//self.slide_factor)+1)*self.window_size
        pbar = tqdm(total=total_sliding_samples)

        window_count = 0
        while stream.has_more_samples():
            if first_run:
                window = stream.next_sample(self.window_size)
                first_run = False
            else:
                add_sample = stream.next_sample(self.slide_factor)
                window_x = np.concatenate([window[0][self.slide_factor:],add_sample[0]],axis=0)
                window_y = np.concatenate([window[1][self.slide_factor:],add_sample[1]],axis=0)
                window = (window_x, window_y)
                
            X, y = window[0], window[1]
            # Test
            if window_count%self.window_wait==0 and len(y)==self.window_size:
                preds = model.predict(X)
                truths = [label for label in y]

                # Metrics
                accuracy = compute_multiclass_accuracy(truths, preds)
                acc_win.append(accuracy*100)

            # Train
            model.partial_fit(X,y)
            # model.partial_fit(X[-1:],y[-1:])
            window_count+=1

            # update the progress bar
            pbar.update(self.window_size)
        
        pbar.close()
        acc = np.array(acc_win).mean()
        return acc_win, acc
